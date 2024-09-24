import os

import scipy
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import ticker
from scipy.stats import gaussian_kde
from torch_geometric.utils import to_networkx


# Function to load the trained model
def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Set the model to evaluation mode


# Function to get predictions from the model
def get_predictions(model, data_loader, device):
    predictions = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            predictions.extend(output.cpu().numpy())
    return predictions


# Function to plot density distributions and compute statistics
def plot_distribution_and_stats(train_predictions, test_predictions):
    # Plot density distributions
    plt.figure(figsize=(12, 6))

    # Plot for training data
    plt.subplot(1, 2, 1)
    sns.kdeplot(train_predictions, shade=True)
    plt.title('Density Plot of Training Predictions')
    plt.xlabel('Predicted Values')
    plt.ylabel('Density')

    # Plot for test data
    plt.subplot(1, 2, 2)
    sns.kdeplot(test_predictions, shade=True)
    plt.title('Density Plot of Test Predictions')
    plt.xlabel('Predicted Values')
    plt.ylabel('Density')

    plt.tight_layout()
    plt.show()

    # Compute statistical characteristics
    train_predictions = np.array(train_predictions)
    test_predictions = np.array(test_predictions)

    train_mean = np.mean(train_predictions)
    train_std_dev = np.std(train_predictions)
    train_median = np.median(train_predictions)

    test_mean = np.mean(test_predictions)
    test_std_dev = np.std(test_predictions)
    test_median = np.median(test_predictions)

    print("Training Data Statistics:")
    print(f"Mean: {train_mean:.4f}")
    print(f"Standard Deviation: {train_std_dev:.4f}")
    print(f"Median: {train_median:.4f}")

    print("\nTest Data Statistics:")
    print(f"Mean: {test_mean:.4f}")
    print(f"Standard Deviation: {test_std_dev:.4f}")
    print(f"Median: {test_median:.4f}")


def plot_deviation_distribution(deviations, epoch):
    if not os.path.exists('testing'):
        os.makedirs('testing')

    try:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(deviations[:, 0], bins=50, alpha=0.75, density=True)
        density = gaussian_kde(deviations[:, 0])
        xs = np.linspace(min(deviations[:, 0]), max(deviations[:, 0]), 200)
        plt.plot(xs, density(xs))
        plt.xlabel(r'Deviation ($y - y_{pred}$)')
        plt.ylabel('Density')
        plt.title(f'Deviation Total Charge - Epoch {epoch}')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.hist(deviations[:, 1], bins=50, alpha=0.75, density=True)
        density = gaussian_kde(deviations[:, 1])
        xs = np.linspace(min(deviations[:, 1]), max(deviations[:, 1]), 200)
        plt.plot(xs, density(xs))
        plt.xlabel(r'Deviation ($y - y_{pred}$)')
        plt.ylabel('Density')
        plt.title(f'Deviation Magnetic Charge - Epoch {epoch}')
        plt.grid(True)

        plt.savefig(f'testing/{epoch}.png')
        plt.close()
    except:
        print('Error to plot Deviation')

    np.savetxt(f'testing/{epoch}.dat', deviations)


def visualize_graph(data):
    """
    Function to visualize the graph using networkx.
    """
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    nx.draw_networkx(G,
                     pos=nx.spring_layout(G, seed=0),
                     with_labels=False,
                     node_size=100,
                     node_color=data.y,
                     cmap="hsv",
                     vmin=-2,
                     vmax=3,
                     width=0.8,
                     edge_color="grey",
                     font_size=14
                     )
    plt.show()


def plot_chgcar(chgcar, name = 'chgcar'):
    chg = np.array(chgcar.data['total'])
    chg_diff = np.array(chgcar.data['diff'])

    chg_2d = np.zeros((chg.shape[0] ** 2, 3))
    for nx in range(chg.shape[0]):
        for ny in range(chg.shape[0]):
            chg_2d[nx * chg.shape[0] + ny, :] = nx / chg.shape[0], ny / chg.shape[0], chg[nx, ny, chg.shape[0] // 2]

    chg_2d_diff = np.zeros((chg_diff.shape[0] ** 2, 3))
    for nx in range(chg_diff.shape[0]):
        for ny in range(chg_diff.shape[0]):
            chg_2d_diff[nx * chg_diff.shape[0] + ny, :] = nx / chg_diff.shape[0], ny / chg_diff.shape[0], chg_diff[
                nx, ny, chg_diff.shape[0] // 2]

    n = 100
    xi_e = np.linspace(0, 1, n)
    yi_e = np.linspace(0, 1, n)
    zi_e = scipy.interpolate.griddata((chg_2d[:, 0], chg_2d[:, 1]), chg_2d[:, 2], (xi_e[None, :], yi_e[:, None]),
                                      method='linear')

    xi_d = np.linspace(0, 1, n)
    yi_d = np.linspace(0, 1, n)
    zi_d = scipy.interpolate.griddata((chg_2d_diff[:, 0], chg_2d_diff[:, 1]), chg_2d_diff[:, 2],
                                      (xi_d[None, :], yi_d[:, None]), method='linear')

    fig, axs = plt.subplots(2, 1, figsize=(20, 10))
    plt.subplots_adjust(hspace=0.10, wspace=0.01)

    cp = axs[0].contourf(xi_e, yi_e, zi_e, cmap='plasma',
                         levels=np.linspace(np.ceil(min(chg_2d[:, 2])), np.ceil(max(chg_2d[:, 2])), 100))
    cbar = fig.colorbar(cp, shrink=0.9, ax=axs[0], pad=0.02,
                        ticks=np.linspace(np.ceil(min(chg_2d[:, 2])), np.ceil(max(chg_2d[:, 2])), 10))
    cbar.ax.set_yticklabels(np.linspace(0, 1, 10))
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(18)
    cbar.ax.set_title(r'/rho', size=20)

    cp = axs[1].contourf(xi_d, yi_d, zi_d, cmap='plasma',
                         levels=np.linspace(np.ceil(min(chg_2d_diff[:, 2])), np.ceil(max(chg_2d_diff[:, 2])), 100))
    cbar = fig.colorbar(cp, shrink=0.9, ax=axs[1], pad=0.02,
                        ticks=np.linspace(np.ceil(min(chg_2d_diff[:, 2])), np.ceil(max(chg_2d_diff[:, 2])), 10))
    '''cbar.ax.set_yticklabels(np.linspace(0, 1, 10))
    for t in cbar.ax.get_yticklabels():
         t.set_fontsize(18)'''
    cbar.ax.set_title(r'/rho', size=20)

    for i in range(2):
        axs[i].tick_params(axis='both',
                           which='major',
                           direction='in',
                           # length = 10,
                           # width = 2,
                           # color = 'm',
                           # pad = 10,
                           labelsize=18,
                           # labelcolor = 'r',
                           bottom=True,
                           top=True,
                           left=True,
                           right=True)

        axs[i].set_xlim(0 + 1 / chg.shape[0], 1 - 1 / chg.shape[0])
        axs[i].set_ylim(0 + 1 / chg.shape[0], 1 - 1 / chg.shape[0])

        axs[i].set_aspect('equal')

        axs[i].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        axs[i].yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    fig.savefig(f'{name}.png', transparent=False, bbox_inches='tight', dpi=300)


#if __name__ == "__main__":
