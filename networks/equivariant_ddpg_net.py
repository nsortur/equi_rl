import torch

from e2cnn import gspaces
from e2cnn import nn

from utils.ddpg_utils import TruncatedNormal


class EquivariantEncoder(torch.nn.Module):
    """
    Equivariant Encoder. The input is a trivial representation with obs_channel channels.
    The output is a regular representation with n_out channels
    """

    def __init__(self, obs_channel=2, n_out=128, initialize=True, N=4):
        super().__init__()
        self.obs_channel = obs_channel
        self.c4_act = gspaces.Rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.c4_act, obs_channel * [self.c4_act.trivial_repr]),
                      nn.FieldType(self.c4_act, n_out//8 * \
                                   [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//8 * \
                    [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(
                self.c4_act, n_out//8 * [self.c4_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//8 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//4 * \
                                   [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//4 * \
                    [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(
                self.c4_act, n_out//4 * [self.c4_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//4 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out//2 * \
                                   [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out//2 * \
                    [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(
                self.c4_act, n_out//2 * [self.c4_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.c4_act, n_out//2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * \
                                   [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * \
                    [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(
                self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out*2 * \
                                   [self.c4_act.regular_repr]),
                      kernel_size=3, padding=1, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out*2 * \
                    [self.c4_act.regular_repr]), inplace=True),

            nn.R2Conv(nn.FieldType(self.c4_act, n_out*2 * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * \
                                   [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * \
                    [self.c4_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(
                self.c4_act, n_out * [self.c4_act.regular_repr]), 2),
            # 3x3
            nn.R2Conv(nn.FieldType(self.c4_act, n_out * [self.c4_act.regular_repr]),
                      nn.FieldType(self.c4_act, n_out * \
                                   [self.c4_act.regular_repr]),
                      kernel_size=3, padding=0, initialize=initialize),
            nn.ReLU(nn.FieldType(self.c4_act, n_out * \
                    [self.c4_act.regular_repr]), inplace=True),
            # 1x1
        )

    def forward(self, geo):
        return self.conv(geo)


class EquivariantDDPGActor(torch.nn.Module):
    """Equivariant DDPG actor network using Wang et al. architecture"""

    def __init__(self, feature_dim=(2, 128, 128), action_shape=5, hidden_dim=128, N=4):
        # TODO: n=2 not implemented because I don't understand n_rho1
        super().__init__()
        self.enc = EquivariantEncoder
        self.r2_rot = gspaces.Rot2dOnR2(N)
        # output of encoder: n_out (aka hidden_dim) * [self.c4_act.regular_repr] (128 channels)
        # n_out of encoder is hidden_dimension of actor network

        self.feature_dim = feature_dim
        self.obs_channels = feature_dim[0]
        self.action_shape = action_shape

        self.network = torch.nn.Sequential(
            self.enc(feature_dim[0], hidden_dim, N=N),
            nn.R2Conv(nn.FieldType(self.r2_rot, hidden_dim * [self.r2_rot.regular_repr]),
                      # TODO: shouldn't this be 1 standard representation for mu of equivariant actions
                      # mixed representation including action_dim trivial representations (for the std of all actions),
                      # (action_dim-2) trivial representations (for the mu of invariant actions),
                      # and 1 standard representation (for the mu of equivariant actions)
                      nn.FieldType(self.r2_rot, [self.r2_rot.irrep(1)] +
                                   (action_shape * 2 - 2) * [self.r2_rot.trivial_repr]),
                      # TODO any reason these kernel sizes (in critic too) are 1 and not 3?
                      kernel_size=1, padding=0)
        )

    def forward(self, obs: torch.Tensor, std: int) -> TruncatedNormal:
        """Returns a distribution over actions for a batch

        Args:
            obs (torch.Tensor): Observation batch of (B, C, H, W), where C, H, W
            must equal feature_dim specified in construction

        Returns:
            TruncatedNormal: A distribution over actions
        """
        assert obs.shape[1:
                         ] == self.feature_dim, f"Observation shape must be {self.feature_dim}, current is {obs.shape[1:]}"
        batch_size = obs.shape[0]

        geo = nn.GeometricTensor(obs, nn.FieldType(self.r2_rot,
                                                   self.obs_channels * [self.r2_rot.trivial_repr]))
        output = self.network(geo).tensor.reshape(batch_size, -1)
        act_xy = output[:, 0:2]
        act_inv = output[:, 2:self.action_shape]
        action = torch.cat((act_inv[:, 0:1], act_xy, act_inv[:, 1:]), dim=1)
        # TODO try standard deviations from here, versus from scheduling
        # if this is worse, remove from networks
        # std = output[:, self.action_shape:]

        dist = TruncatedNormal(action, std)
        return dist


class EquivariantDDPGCritic(torch.nn.Module):
    """Equivariant DDPG double critic networks using Wang et al. architecture"""

    def __init__(self, feature_dim=(2, 128, 128), action_shape=5, hidden_dim=128, N=4):
        super().__init__()
        self.r2_rot = gspaces.Rot2dOnR2(N)

        self.feature_dim = feature_dim
        self.action_shape = action_shape
        self.hidden_dim = hidden_dim
        self.obs_channels = feature_dim[0]

        # encoder must be separate so we can get state information, then pass into critics
        self.encoder = EquivariantEncoder(feature_dim[0], hidden_dim, N=N)
        self.critic1 = torch.nn.Sequential(
            # mixed representation including n_hidden regular representations (for the state),
            # (action_dim-2) trivial representations (for the invariant actions)
            # and 1 standard representation (for the equivariant actions)
            nn.R2Conv(nn.FieldType(self.r2_rot, hidden_dim * [self.r2_rot.regular_repr] +
                                   (action_shape-2) * [self.r2_rot.trivial_repr] +
                                   [self.r2_rot.irrep(1)]),
                      nn.FieldType(self.r2_rot, hidden_dim * \
                                   [self.r2_rot.regular_repr]),
                      kernel_size=1, padding=0),
            nn.ReLU(nn.FieldType(self.r2_rot, hidden_dim * [self.r2_rot.regular_repr]),
                    inplace=True),
            nn.GroupPooling(nn.FieldType(
                self.r2_rot, hidden_dim * [self.r2_rot.regular_repr])),
            nn.R2Conv(nn.FieldType(self.r2_rot, hidden_dim * [self.r2_rot.trivial_repr]),
                      nn.FieldType(self.r2_rot, 1 * \
                                   [self.r2_rot.trivial_repr]),
                      kernel_size=1, padding=0)
        )
        self.critic2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.r2_rot, hidden_dim * [self.r2_rot.regular_repr] +
                                   (action_shape-2) * [self.r2_rot.trivial_repr] +
                                   [self.r2_rot.irrep(1)]),
                      nn.FieldType(self.r2_rot, hidden_dim *
                                   [self.r2_rot.regular_repr]),
                      kernel_size=1, padding=0),
            nn.ReLU(nn.FieldType(self.r2_rot, hidden_dim * [self.r2_rot.regular_repr]),
                    inplace=True),
            nn.GroupPooling(nn.FieldType(
                self.r2_rot, hidden_dim * [self.r2_rot.regular_repr])),
            nn.R2Conv(nn.FieldType(self.r2_rot, hidden_dim * [self.r2_rot.trivial_repr]),
                      nn.FieldType(self.r2_rot, 1 *
                                   [self.r2_rot.trivial_repr]),
                      kernel_size=1, padding=0)
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        """Evaluates estimated q-value of state, action pair

        Args:
            obs (_type_): Image observation in the same shape as feature_dim
            act (_type_): Action with same dimensions as action_shape

        Returns:
            tuple(torch.Tensor, torch.Tensor): Q-values for the first and second critic
        """
        assert obs.shape[1:
                         ] == self.feature_dim, f"Observation shape must be {self.feature_dim}, current is {obs.shape[1:]}"
        assert act.shape[1:][0] == self.action_shape, f"Action shape must be {self.action_shape}, current is {act.shape[1:][0]}"
        batch_size = obs.shape[0]

        # separate action into into invariant and equivariant parts
        dxy = act[:, 1:3]
        inv_act = torch.cat((act[:, 0:1], act[:, 3:]), dim=1)
        n_inv = inv_act.shape[1]

        # TODO: move creating geo to encoder for more consistent code (caller just specifies
        # non-geometric tensor)
        obs_geo = nn.GeometricTensor(obs, nn.FieldType(self.r2_rot,
                                                       self.obs_channels * [self.r2_rot.trivial_repr]))
        obs_encoded = self.encoder(obs_geo)
        cat = torch.cat((obs_encoded.tensor, inv_act.reshape(
            batch_size, n_inv, 1, 1), dxy.reshape(batch_size, 2, 1, 1)), dim=1)
        obs_act_geo = nn.GeometricTensor(cat,
                                         # output of encoder, invariant actions, and xy equivariant actions
                                         nn.FieldType(self.r2_rot, self.hidden_dim * [self.r2_rot.regular_repr] +
                                                      n_inv * [self.r2_rot.trivial_repr] +
                                                      [self.r2_rot.irrep(1)]))

        out1 = self.critic1(obs_act_geo).tensor.reshape(batch_size, 1)
        out2 = self.critic1(obs_act_geo).tensor.reshape(batch_size, 1)
        return out1, out2
