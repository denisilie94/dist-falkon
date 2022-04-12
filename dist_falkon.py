import time
import scipy.io
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.linalg import solve_triangular
from sklearn.metrics.pairwise import rbf_kernel


def _preconditioner(X_m, kernel_method, _lambda, _gamma):
    m = X_m.shape[0]
    K_mm = kernel_method(X_m, X_m, _gamma)

    # for i in range(m):
    #       K_mm[i, i] += 1e-12

    T = np.linalg.cholesky(K_mm).T
    K_flipped = np.dot(T, T.T) / m
    for i in range(m):
        K_flipped[i, i] += _lambda
    A = np.linalg.cholesky(K_flipped).T
    return T, A


class Server(multiprocessing.Process):
    def __init__(self, _n_centroids, _n_iter, _n_clients, clients_pipes, save_to_npz=False):
        multiprocessing.Process.__init__(self)

        self._n_centroids = _n_centroids
        self._n_iter = _n_iter
        self._n_clients = _n_clients
        self.clients_pipes = clients_pipes
        self.beta = None
        self.betas = None
        self.safe_to_npz = save_to_npz

    def run(self) -> None:
        for i_iter in tqdm(range(self._n_iter)):
            self.beta = np.zeros((self._n_centroids, 1))
            self.betas = []

            # receive results from all clients
            for client_pipe in self.clients_pipes:
                beta = client_pipe.recv()
                self.betas.append(beta)
                self.beta += beta / self._n_clients

            # send computed solution on server
            for client_pipe in self.clients_pipes:
                client_pipe.send(self.beta)

            # save server result to npz
            if self.safe_to_npz:
                np.savez('./npzs/beta_server_iter_{}.npz'.format(i_iter), beta=self.beta)


class Client(multiprocessing.Process):
    def __init__(self, X, y, X_m, A, T, _lambda, _gamma, _n_iter,
                 kernel_method, server_pipe, client_index, save_to_npz=False):
        multiprocessing.Process.__init__(self)

        self.X = X
        self.y = y
        self.X_m = X_m
        self.A = A
        self.T = T

        self._lambda = _lambda
        self._gamma = _gamma
        self._n_iter = _n_iter
        self.kernel_method = kernel_method
        self.server_pipe = server_pipe

        self.n = X.shape[0]
        self.beta = None
        self.K_m = None
        self.r = None

        self.client_index = client_index
        self.save_to_npz = save_to_npz

    def lin_op(self, beta):
        v = solve_triangular(self.A, beta, lower=False)
        c = np.dot(self.K_m, np.dot(self.K_m.T, solve_triangular(self.T, v, lower=False))) / np.array(self.n)
        return solve_triangular(
            self.A.T,
            solve_triangular(self.T.T, c, lower=True) + self._lambda * v,
            lower=True
        )

    def conjgrad(self, x, max_iter):
        x = np.zeros((self.A.shape[1], 1)) if x is None else x
        residual = self.r - self.lin_op(x)
        p = residual
        rsold = np.dot(np.transpose(residual), residual)

        for i_iter in range(max_iter):
            Ap = self.lin_op(p)
            alpha = rsold / np.dot(np.transpose(p), Ap)
            x = x + alpha * p
            residual = residual - alpha * Ap
            rsnew = np.dot(np.transpose(residual), residual)
            p = residual + (rsnew / rsold) * p
            rsold = rsnew
        return x

    def run(self) -> None:
        # compute kernel matrix
        self.K_m = self.kernel_method(self.X_m, self.X, self._gamma)

        # compute r
        self.r = solve_triangular(self.T.T, np.dot(self.K_m, self.y / len(self.y)), lower=True)
        self.r = solve_triangular(self.A.T, self.r, lower=True)

        for i_iter in range(self._n_iter):
            # compute conjugate gradient
            self.beta = self.conjgrad(self.beta, 1)

            # send client solution
            self.server_pipe.send(self.beta)

            if self.save_to_npz:
                np.savez('./npzs/beta_client_{}_iter_{}.npz'.format(self.client_index, i_iter), beta=self.beta)

            # receive the server solution
            self.beta = self.server_pipe.recv()


if __name__ == '__main__':
    # dataset = 'susy.mat'
    dataset = 'million_songs.mat'
    # dataset = 'million_songs_50k.mat'

    if dataset == 'susy.mat':
        _lambda = 1e-6    # lambda
        _n_iter = 20      # number of rounds
        _gamma = 0.03125  # kernel parameter
    elif dataset == 'million_songs.mat' or dataset == 'million_songs_50k.mat':
        _lambda = 1e-6    # lambda
        _n_iter = 20      # number of rounds
        _gamma = 0.01388  # kernel parameter
    else:
        raise Exception('Unknown dataset')

    mat = scipy.io.loadmat('./data/' + dataset)

    X_train = mat['Xtr']
    X_test = mat['Xts']
    X_m = mat['Xuni']
    M = X_m.shape[0]

    y_train = mat['Ytr']
    y_test = mat['Yts']
    y_tr0 = mat['Ytr0'] if 'Ytr0' in mat else None

    # get original data
    X_train_origin = X_train[:100000, :]
    y_train_origin = y_train[:100000]

    # problem params
    _n_clients = 5
    _compare_results = True
    kernel_method = rbf_kernel

    # split data for clients
    X_train_list = np.split(X_train_origin, _n_clients)
    y_train_list = np.split(y_train_origin, _n_clients)

    m = 20000
    X_m = X_m[:m, :]
    server_pipes = []
    clients_list = []
    clients_pipes = []

    # run problem
    t0 = time.time()
    T, A = _preconditioner(X_m, rbf_kernel, _lambda, _gamma)
    preconditioning_time = time.time() - t0

    # init pipes
    for i_client in range(_n_clients):
        s_pipe, c_pipe = multiprocessing.Pipe()
        server_pipes.append(s_pipe)
        clients_pipes.append(c_pipe)

    # init clients
    for i_client in range(_n_clients):
        client = Client(
            X_train_list[i_client],
            y_train_list[i_client],
            X_m,
            A,
            T,
            _lambda,
            _gamma,
            _n_iter,
            kernel_method,
            clients_pipes[i_client],
            i_client,
            True
        )

        clients_list.append(client)
        clients_list[i_client].start()

    # init server
    server = Server(X_m.shape[0], _n_iter, _n_clients, server_pipes, True)
    server.start()

    t0 = time.time()

    # join all processes
    for i_client in range(_n_clients):
        clients_list[i_client].join()
    server.join()

    tf = time.time() - t0
    delta_t = tf + preconditioning_time

    # read server result
    data = np.load('./npzs/beta_server_iter_{}.npz'.format(_n_iter-1))
    beta = data['beta']

    # compute alpha solution
    alpha = solve_triangular(A, beta, lower=False)
    alpha = solve_triangular(T, alpha, lower=False)

    # compute tests
    K_m_test = kernel_method(X_m, X_test, _gamma)
    RMSE = np.linalg.norm(y_test - np.dot(K_m_test.T, alpha)) / np.sqrt(len(y_test))

    def compute_mse(alpha=None, K_m_test=K_m_test, y_test=y_test, y_tr0=y_tr0):
        y_pred = np.dot(K_m_test.T, alpha)
        if y_tr0 is not None:
            y_tr0_std = np.std(y_tr0)
            MSE = np.mean(np.power(y_pred*y_tr0_std - y_test*y_tr0_std, 2))
        else:
            MSE = np.mean(np.power(y_test - y_pred, 2))
        return MSE

    MSE = compute_mse(alpha, K_m_test, y_test, y_tr0)

    if _compare_results:
        plt.figure()
        plt.xlabel('iter')
        plt.ylabel('log2(rmse)')
        plt.title('RMSE evolution - iter')

        for i_client in range(_n_clients):
            rmse = []

            for i_iter in range(_n_iter):
                data = np.load('./npzs/beta_client_{}_iter_{}.npz'.format(i_client, i_iter))
                beta = data['beta']

                # compute alpha solution
                alpha = solve_triangular(A, beta, lower=False)
                alpha = solve_triangular(T, alpha, lower=False)

                # compute RMSE
                RMSE = np.linalg.norm(y_test - np.dot(K_m_test.T, alpha)) / np.sqrt(len(y_test))
                rmse.append(RMSE)

            plt.plot(range(len(rmse)), np.log2(rmse), label='client ' + str(i_client))

        # load server solution
        rmse = []

        for i_iter in range(_n_iter):
            data = np.load('./npzs/beta_server_iter_{}.npz'.format(i_iter))
            beta = data['beta']

            # compute alpha solution
            alpha = solve_triangular(A, beta, lower=False)
            alpha = solve_triangular(T, alpha, lower=False)

            # compute RMSE
            RMSE = np.linalg.norm(y_test - np.dot(K_m_test.T, alpha)) / np.sqrt(len(y_test))
            rmse.append(RMSE)
        
        plt.plot(range(len(rmse)), np.log2(rmse), label='server')    
        
        # load origin solution
        rmse = []
        
        for i_iter in range(_n_iter):
            data = np.load('./npzs/beta_origin_iter_{}.npz'.format(i_iter))
            beta = data['beta']
            
            # compute alpha solution
            alpha = solve_triangular(A, beta, lower=False)
            alpha = solve_triangular(T, alpha, lower=False)

            # compute RMSE
            RMSE = np.linalg.norm(y_test - np.dot(K_m_test.T, alpha)) / np.sqrt(len(y_test))
            rmse.append(RMSE)

        plt.plot(range(len(rmse)), np.log2(rmse), label='origin')
        
        plt.show()
        plt.legend()
