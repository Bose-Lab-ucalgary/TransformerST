# Based on https://github.com/kasparmartens/BasisVAE/blob/master/BasisVAE/decoder.py
# Originally authored by Kaspar Märtens
# Significant added components: StandardDecoder, ODEDecoder, BasisODEDecoder

import numpy as np

import torch
import torch.nn as nn

from torch.nn.functional import softplus
from torch.nn import functional as F

# from torch.distributions.uniform import Uniform
from torch.distributions.gamma import Gamma
from torch.distributions.dirichlet import Dirichlet

from src.Utils.helpers import NB_log_prob, ZINB_log_prob, ELBO_collapsed_Categorical, init_weights

class StandardDecoder(nn.Module):
    """Standard Neural Network decoder of a standard VAE."""
    def __init__(self, data_dim, hidden_dim, z_dim,
                 likelihood="Gaussian",
                 nonlinearity=nn.Softplus,
                 device="cpu"):
        super().__init__()

        self.data_dim = data_dim
        self.likelihood = likelihood
        if device == "gpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Decoder: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"Decoder: {device} specified, {self.device} used")

        self.mapping_z = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nonlinearity(),
            nn.Linear(hidden_dim, data_dim)
        )

        if self.likelihood == "Gaussian":
            self.noise_sd = nn.Parameter(-2.0 * torch.ones(1, data_dim))

        elif self.likelihood == "NB":
            self.nb_theta = nn.Parameter(torch.zeros(1, data_dim))

        elif self.likelihood == "ZINB":
            self.nb_theta = nn.Parameter(torch.zeros(1, data_dim))

            self.dropout_decoder = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nonlinearity(),
                nn.Linear(hidden_dim, data_dim)
            )

        elif self.likelihood == "Bernoulli":
            self.noise_sd = None
        else:
            raise ValueError("Unknown likelihood")

    def forward(self, z):
        if self.likelihood == "Bernoulli":
            pred = self.mapping_z(z)
            dropout_prob = None
            theta = None

        elif self.likelihood == "Gaussian":
            pred = self.mapping_z(z)
            dropout_prob = None
            theta = None

        elif self.likelihood == "NB":
            pred = softplus(self.mapping_z(z))
            theta = softplus(self.nb_theta)
            dropout_prob = None

        elif self.likelihood == "ZINB":
            pred = softplus(self.mapping_z(z))
            dropout_prob = self.dropout_decoder(z)
            theta = softplus(self.nb_theta)

        return pred[:,:,None], dropout_prob, theta

    def loglik(self, y_obs, y_pred, dropout_prob_logit, theta):

        if self.likelihood == "Gaussian":

            sigma = 1e-4 + softplus(self.noise_sd)
            p_data = torch.distributions.normal.Normal(loc=y_pred, scale=sigma[:, :, None])
            log_p = p_data.log_prob(y_obs[:, :, None])

        elif self.likelihood == "NB":

            log_p = NB_log_prob(y_obs[:, :, None], mu=y_pred, theta=theta[:, :, None])

        elif self.likelihood == "ZINB":

            log_p = ZINB_log_prob(y_obs[:, :, None], y_pred, theta[:, :, None], dropout_prob_logit[:, :, None])

        if self.likelihood == "Bernoulli":

            log_p = -F.binary_cross_entropy_with_logits(y_pred, y_obs[:, :, None].repeat(1, 1, self.n_basis), reduction='none')

        loglik = log_p.sum()

        return loglik

    def loss(self, y_obs, y_pred, dropout_prob_logit, theta, batch_scale=1.0, beta=1.0, prior_loss=True):

        loglik = batch_scale * self.loglik(y_obs, y_pred, dropout_prob_logit, theta)

        decoder_loss = - loglik

        return decoder_loss


class ODEDecoder(nn.Module):
    """Derivative-based decoder of DeVAE."""
    def __init__(self, data_dim, hidden_dim, z_dim,
                 likelihood="Gaussian",
                 nonlinearity=nn.Softplus,
                 device="cpu",
                 n=15, x0init=None):
        super().__init__()

        self.data_dim = data_dim
        self.likelihood = likelihood
        if device == "gpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Decoder: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"Decoder: {device} specified, {self.device} used")

        self.output_dim = data_dim

        self.dudt = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nonlinearity(),
            nn.Linear(hidden_dim, self.output_dim)#,
            # nn.Softplus()
        )

        # feature-specific variances
        if self.likelihood == "Gaussian":
            self.noise_sd = nn.Parameter(-2.0 * torch.ones(1, data_dim))

        elif self.likelihood == "NB":
            self.nb_theta = nn.Parameter(torch.zeros(1, data_dim))

        elif self.likelihood == "ZINB":
            self.nb_theta = nn.Parameter(torch.zeros(1, data_dim))

            self.dropout_decoder = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nonlinearity(),
                nn.Linear(hidden_dim, data_dim)
            )

        elif self.likelihood == "Bernoulli":
            self.noise_sd = None
        else:
            raise ValueError("Unknown likelihood")

        if x0init:
            self.x0 = nn.Parameter(torch.tensor(x0init))
        else:
            self.x0 = nn.Parameter(torch.zeros(1, data_dim))

        self.n = n
        u_n, w_n = np.polynomial.legendre.leggauss(n)
        self.u_n = nn.Parameter(torch.tensor(u_n,device=self.device,dtype=torch.float32)[None,:],requires_grad=False)
        self.w_n = nn.Parameter(torch.tensor(w_n,device=self.device,dtype=torch.float32)[None,:],requires_grad=False)

    def get_x0(self):
        return self.x0

    def mapping_z(self, z):
        f_arg = torch.matmul(z/2, 1+self.u_n)
        f_n = self.dudt(torch.flatten(f_arg)[:,None]).reshape((*f_arg.shape,self.output_dim))
        pred = self.get_x0() + z/2 * (self.w_n[:,:,None] * f_n).sum(dim=1)
        return pred

    def forward(self, z):
        if self.likelihood == "Bernoulli":
            pred = self.mapping_z(z)
            dropout_prob = None
            theta = None

        elif self.likelihood == "Gaussian":
            pred = self.mapping_z(z)
            dropout_prob = None
            theta = None

        elif self.likelihood == "NB":
            pred = softplus(self.mapping_z(z))
            theta = softplus(self.nb_theta)
            dropout_prob = None

        elif self.likelihood == "ZINB":
            pred = softplus(self.mapping_z(z))
            dropout_prob = self.dropout_decoder(z)
            theta = softplus(self.nb_theta)

        return pred[:,:,None], dropout_prob, theta

    def loglik(self, y_obs, y_pred, dropout_prob_logit, theta):

        if self.likelihood == "Gaussian":

            sigma = 1e-4 + softplus(self.noise_sd)
            p_data = torch.distributions.normal.Normal(loc=y_pred, scale=sigma[:, :, None])
            log_p = p_data.log_prob(y_obs[:, :, None])

        elif self.likelihood == "NB":

            log_p = NB_log_prob(y_obs[:, :, None], mu=y_pred, theta=theta[:, :, None])

        elif self.likelihood == "ZINB":

            log_p = ZINB_log_prob(y_obs[:, :, None], y_pred, theta[:, :, None], dropout_prob_logit[:, :, None])

        if self.likelihood == "Bernoulli":

            log_p = -F.binary_cross_entropy_with_logits(y_pred, y_obs[:, :, None].repeat(1, 1, self.n_basis), reduction='none')

        loglik = log_p.sum()

        return loglik

    def loss(self, y_obs, y_pred, dropout_prob_logit, theta, batch_scale=1.0, beta=1.0, prior_loss=True):

        loglik = batch_scale * self.loglik(y_obs, y_pred, dropout_prob_logit, theta)

        decoder_loss = - loglik

        return decoder_loss


class BasisODEDecoder(nn.Module):
    """Derivative-based decoder with feature-level clustering: BasisDeVAE."""
    def __init__(self, data_dim, hidden_dim, z_dim,
                 n_basis,
                 likelihood="Gaussian",
                 nonlinearity=nn.Softplus,
                 alpha=1.0,
                 device="cpu",
                 n=15, x0init=None):
        super().__init__()

        self.data_dim = data_dim
        self.likelihood = likelihood
        if device == "gpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Decoder: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"Decoder: {device} specified, {self.device} used")

        self.n_basis = n_basis
        self.output_dim = data_dim * n_basis

        self.linear1 = nn.Linear(z_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.output_dim)
        self.nonlinearity = nonlinearity()
        self.base = nn.Sequential(self.linear1,
                                self.nonlinearity,
                                self.linear2,
                                nn.Softplus()).apply(init_weights)

        # feature-specific variances
        if self.likelihood == "Gaussian":
            self.noise_sd = nn.Parameter(-2.0 * torch.ones(1, data_dim))

        elif self.likelihood == "NB":
            self.nb_theta = nn.Parameter(torch.zeros(1, data_dim))

        elif self.likelihood == "ZINB":
            self.nb_theta = nn.Parameter(torch.zeros(1, data_dim))

            self.dropout_decoder = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nonlinearity(),
                nn.Linear(hidden_dim, data_dim)
            )

        elif self.likelihood == "Bernoulli":
            self.noise_sd = None
        else:
            raise ValueError("Unknown likelihood")

        self.alpha_z = alpha * torch.ones(n_basis, device=self.device)

        self.qphi_logits = nn.Parameter(torch.ones([self.data_dim, self.n_basis]))

        self.scaling_z = nn.Parameter(torch.zeros([self.data_dim, 1]))
        self.shift_z = nn.Parameter(0.0*torch.ones([self.data_dim, 1, 1]))

        if x0init is not None:
            self.x0 = nn.Parameter(torch.tensor(x0init))
        else:
            self.x0 = nn.Parameter(2 * torch.ones(1, data_dim))

        self.gaussx0 = nn.Parameter(torch.zeros(1,data_dim))

        self.n = n
        u_n, w_n = np.polynomial.legendre.leggauss(n)
        self.u_n = nn.Parameter(torch.tensor(u_n,device=self.device,dtype=torch.float32)[None,:],requires_grad=False)
        self.w_n = nn.Parameter(torch.tensor(w_n,device=self.device,dtype=torch.float32)[None,:],requires_grad=False)

    def get_delta(self):
        return self.shift_z

    def get_lambda(self):
        return 1e-4 + softplus(self.scaling_z)

    def get_x0(self):
        return self.x0

    def get_gaussx0(self):
        return self.gaussx0

    def get_phi(self):
        return torch.softmax(self.qphi_logits, dim=-1)

    def dudt(self,z):
        dudt_ = self.base(z)
        z_tilde = z - torch.squeeze(self.get_delta())[None,:]
        return torch.cat((dudt_[:,:self.data_dim],-dudt_[:,self.data_dim:2*self.data_dim],dudt_[:,-self.data_dim:] * z_tilde),1)

    def mapping_z(self, z):
        # print(self.u_n.shape,z.shape)
        f_arg = torch.matmul(z/2, 1+self.u_n)
        f_n = self.dudt(torch.flatten(f_arg)[:,None]).reshape((*f_arg.shape,self.output_dim))
        pred = z/2 * (self.w_n[:,:,None] * f_n).sum(dim=1)
        return pred

    def get_basis(self, z):
        basis0 = self.mapping_z(z).reshape((z.shape[0],self.n_basis,self.data_dim))
        basis0 = torch.einsum('ijk -> ikj', basis0)
        basis = self.get_x0()[:,:,None] + basis0
        z_tilde = z - torch.squeeze(self.get_delta())[None,:]
        gauss_basis = softplus(basis0[:,:,2] + self.get_gaussx0()) * z_tilde
        return torch.cat((basis[:,:,:2],self.get_lambda()[None,:,:] * torch.exp(-gauss_basis[:,:,None] ** 2)),2)

    def KL_phi(self):
        return ELBO_collapsed_Categorical(self.qphi_logits, self.alpha_z, K=self.n_basis, N=self.data_dim)

    def forward(self, z):
        if self.likelihood == "Bernoulli":
            pred = self.get_basis(z)
            dropout_prob = None
            theta = None

        elif self.likelihood == "Gaussian":
            pred = self.get_basis(z)
            dropout_prob = None
            theta = None

        elif self.likelihood == "NB":
            basis = self.get_basis(z)
            pred = torch.cat((softplus(basis[:,:,:2]), basis[:,:,2]))
            theta = softplus(self.nb_theta)
            dropout_prob = None

        elif self.likelihood == "ZINB":
            basis = self.get_basis(z)
            pred = torch.cat((softplus(basis[:,:,:2]), basis[:,:,2][:,:,None]),2)
            theta = softplus(self.nb_theta)
            dropout_prob = self.dropout_decoder(z)

        return pred, dropout_prob, theta

    def loglik(self, y_obs, y_pred, dropout_prob_logit, theta):

        if self.likelihood == "Gaussian":

            sigma = 1e-4 + softplus(self.noise_sd)
            p_data = torch.distributions.normal.Normal(loc=y_pred, scale=sigma[:, :, None])
            log_p = p_data.log_prob(y_obs[:, :, None])

        elif self.likelihood == "NB":

            log_p = NB_log_prob(y_obs[:, :, None], mu=y_pred, theta=theta[:, :, None])

        elif self.likelihood == "ZINB":

            # [batch_size, output_dim, n_basis]
            log_p = ZINB_log_prob(y_obs[:, :, None], y_pred, theta[:, :, None], dropout_prob_logit[:, :, None])

        if self.likelihood == "Bernoulli":

            log_p = -F.binary_cross_entropy_with_logits(y_pred, y_obs[:, :, None].repeat(1, 1, self.n_basis), reduction='none')

        phi = self.get_phi()

        loglik = (phi * log_p).mean()

        return loglik

    def loss(self, y_obs, y_pred, dropout_prob_logit, theta, batch_scale=1.0, beta=1.0, prior_loss=True):

        prior_loss = 0.0

        prior_loss += self.get_delta().pow(2).mean()

        lambdas = self.get_lambda()
        prior_loss -= Gamma(torch.ones_like(lambdas), torch.ones_like(lambdas)).log_prob(lambdas).mean()

        loglik = batch_scale * self.loglik(y_obs, y_pred, dropout_prob_logit, theta)

        decoder_loss = - loglik + beta * self.KL_phi()
        if prior_loss:
            decoder_loss = decoder_loss + prior_loss

        return decoder_loss


class BasisDecoder(nn.Module):

    """
    Core component of BasisVAE: decoder where the last layer contains probabilistic Categorical random variables
    """

    def __init__(self, data_dim, hidden_dim, z_dim,
                 n_basis,
                 scale_invariance,
                 translation_invariance,
                 likelihood="Gaussian",
                 nonlinearity=nn.Softplus,
                 inference = "collapsed",
                 alpha=1.0,
                 qalpha_init=None,
                 max_delta=1.5,
                 min_lambda=0.25,
                 max_lambda=1.75,
                 device="gpu"):
        """
        :param data_dim: Data dimensionality
        :param hidden_dim: The number of neurons in the hidden layer (we assume one-hidden-layer NN)
        :param z_dim: Dimensionality of latent space
        :param n_basis: Number of basis functions (K)
        :param scale_invariance: Is BasisVAE scale-invariant?
        :param translation_invariance: Is BasisVAE translation-invariant?
        :param likelihood: Likelihood (options include "Gaussian", "Bernoulli", and for single-cell applications "NB" and "ZINB")
        :param nonlinearity: Type of non-linearity in NNs
        :param inference: Inference approach ("collapsed" is recommended)
        :param alpha: The Dirichlet alpha parameter (scalar)
        :param qalpha_init: Only relevant for non-collapsed inference
        :param max_delta: The range of delta values can be restricted for identifiability
        :param min_lambda: Lower bound on lambda values
        :param max_lambda: Upper bound on lambda values
        :param device: CPU or GPU
        """
        super().__init__()

        self.data_dim = data_dim
        self.likelihood = likelihood
        self.inference = inference

        if device == "gpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"Decoder: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"Decoder: {device} specified, {self.device} used")

        self.scale_invariance = scale_invariance
        self.translation_invariance = translation_invariance
        self.n_basis = n_basis
        self.max_delta = max_delta
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda

        # we will set up a neural network with one hidden layer
        if self.translation_invariance:
            # for translation-invariant case, we do the computations manually for computational efficiency
            self.linear1 = nn.Linear(z_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, self.n_basis)
            self.nonlinearity = nonlinearity()
        else:
            # for non-translation-invariant case
            self.mapping_z = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nonlinearity(),
                nn.Linear(hidden_dim, self.n_basis)
            )

        # feature-specific variances
        if self.likelihood == "Gaussian":
            self.noise_sd = nn.Parameter(-2.0 * torch.ones(1, data_dim))

        elif self.likelihood == "NB":
            self.nb_theta = nn.Parameter(torch.zeros(1, data_dim))

        elif self.likelihood == "ZINB":
            self.nb_theta = nn.Parameter(torch.zeros(1, data_dim))

            self.dropout_decoder = nn.Sequential(
                nn.Linear(z_dim, hidden_dim),
                nonlinearity(),
                nn.Linear(hidden_dim, data_dim)
            )

        elif self.likelihood == "Bernoulli":
            self.noise_sd = None
        else:
            raise ValueError("Unknown likelihood")

        self.intercept = nn.Parameter(torch.zeros(1, data_dim))

        # we assume vector (alpha, ..., alpha)
        self.alpha_z = alpha * torch.ones(n_basis, device=self.device)

        # q(phi) parameters
        self.qphi_logits = nn.Parameter(torch.zeros([self.data_dim, self.n_basis]))

        if self.inference == "non-collapsed":
            # for non-collapsed inference, q(alpha)
            if qalpha_init is None:
                raise ValueError("For non-collapsed inference need to specify q(alpha)")
            self.qalpha_z = nn.Parameter(torch.Tensor(qalpha_init))

        if self.scale_invariance:
            self.scaling_z = nn.Parameter(torch.zeros([self.data_dim, self.n_basis]))

        if self.translation_invariance:
            self.shift_z = nn.Parameter(torch.zeros([self.data_dim, self.n_basis, 1]))

    def get_delta(self):
        # delta values (constrained within [-max_delta, max_delta]
        return self.max_delta * torch.tanh(self.shift_z)

    def get_lambda(self):
        # lambda values
        return self.min_lambda + (self.max_lambda - self.min_lambda) * torch.sigmoid(self.scaling_z)

    def get_phi(self):
        return torch.softmax(self.qphi_logits, dim=-1)

    def get_basis(self, z):

        if self.translation_invariance:

            z_tilde = self.get_delta()[None, :, :, :] + z[:, None, None, :]
            # first hidden layer representation, [N, output_dim, n_basis_z, n_hidden_units]
            hidden = self.nonlinearity(self.linear1(z_tilde))

            # [N, output_dim, n_basis_z]
            basis0 = torch.sum(hidden * self.linear2.weight, dim=3)

            # [output_dim, n_basis_z]
            scaling = self.get_lambda()

            # [N, output_dim, n_basis_z]
            basis = basis0 * scaling

        else:
            basis0 = self.mapping_z(z)

            if self.scale_invariance:
                # shape [output_dim, n_basis_z]
                scaling = self.get_lambda()

                # shape [N, output_dim, n_basis_z]
                basis = basis0[:, None, :] * scaling
            else:
                # shape [N, output_dim, n_basis_z]
                basis = basis0[:, None, :].repeat(1, self.data_dim, 1)

        return basis

    def KL_phi(self):

        if self.inference == "collapsed":
            return ELBO_collapsed_Categorical(self.qphi_logits, self.alpha_z, K=self.n_basis, N=self.data_dim)

        elif self.inference == "fixed_pi":
            qphi = self.get_phi()
            pi = torch.ones_like(qphi) / self.n_basis
            KL = (
                qphi * (torch.log(qphi + 1e-16) - torch.log(pi))
            ).sum()
            return KL

        elif self.inference == "non-collapsed":
            qDir = Dirichlet(concentration=self.qalpha_z)
            pDir = Dirichlet(concentration=self.alpha_z)

            # KL(q(pi) || p(pi))
            KL_Dir = torch.distributions.kl_divergence(qDir, pDir)

            # E[log q(phi) - log p(phi | pi)] under q(pi)q(phi)
            qpi = qDir.rsample()
            qphi = self.get_phi()

            # KL categorical
            KL_Cat = (
                    qphi * (torch.log(qphi + 1e-16) - torch.log(qpi[None, :]))
            ).sum()
            return KL_Dir + KL_Cat

    def forward(self, z):
        if self.likelihood == "Bernoulli":
            pred = self.get_basis(z)
            dropout_prob = None
            theta = None

        elif self.likelihood == "Gaussian":
            pred = self.get_basis(z)
            dropout_prob = None
            theta = None

        elif self.likelihood == "NB":
            pred = softplus(self.get_basis(z))
            theta = softplus(self.nb_theta)
            dropout_prob = None

        elif self.likelihood == "ZINB":
            pred = softplus(self.get_basis(z))
            dropout_prob = self.dropout_decoder(z)
            theta = softplus(self.nb_theta)

        return pred, dropout_prob, theta

    def loglik(self, y_obs, y_pred, dropout_prob_logit, theta):

        if self.likelihood == "Gaussian":

            sigma = 1e-4 + softplus(self.noise_sd)
            p_data = torch.distributions.normal.Normal(loc=y_pred, scale=sigma[:, :, None])
            log_p = p_data.log_prob(y_obs[:, :, None])

        elif self.likelihood == "NB":

            log_p = NB_log_prob(y_obs[:, :, None], mu=y_pred, theta=theta[:, :, None])

        elif self.likelihood == "ZINB":

            # [batch_size, output_dim, n_basis]
            log_p = ZINB_log_prob(y_obs[:, :, None], y_pred, theta[:, :, None], dropout_prob_logit[:, :, None])

        if self.likelihood == "Bernoulli":

            log_p = -F.binary_cross_entropy_with_logits(y_pred, y_obs[:, :, None].repeat(1, 1, self.n_basis), reduction='none')

        phi = self.get_phi()

        loglik = (phi * log_p).mean()

        return loglik

    def loss(self, y_obs, y_pred, dropout_prob_logit, theta, batch_scale=1.0, beta=1.0):

        prior_loss = 0.0

        if self.translation_invariance:
            prior_loss += self.get_delta().pow(2).mean()

        if self.scale_invariance:
            lambdas = self.get_lambda()
            prior_loss -= Gamma(torch.ones_like(lambdas), torch.ones_like(lambdas),validate_args=False).log_prob(lambdas).mean()

        loglik = batch_scale * self.loglik(y_obs, y_pred, dropout_prob_logit, theta)

        decoder_loss = - loglik + beta * self.KL_phi() + prior_loss

        return decoder_loss

    def get_similarity_matrix(self):
        with torch.no_grad():
            mat = torch.mm(self.get_phi(), self.get_phi().t())
        return mat
