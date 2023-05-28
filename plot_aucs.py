import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    epsilons = np.arange(0.05, 1.05, 0.05).tolist()

    colors = ["red", "blue", "green", "black", "cyan", "magenta"]

    # iris = {0.05: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.1: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.15000000000000002: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.2: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.25: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.3: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.35000000000000003: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.4: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.45: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.5: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.55: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.6000000000000001: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.6500000000000001: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.7000000000000001: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.7500000000000001: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.8: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.8500000000000001: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.9000000000000001: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 0.9500000000000001: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}, 1.0: {'Iris-versicolor': 0.95, 'Iris-virginica': 0.975, 'Iris-setosa': 1.0}}
    ive = [0.95] * 20
    ivi = [0.975] * 20
    ise = [1.0] * 20
    breast_w = [0.9465579710144929, 0.9519927536231885, 0.9519927536231885, 0.9519927536231885, 0.9519927536231885, 0.9519927536231885, 0.9519927536231885, 0.9519927536231885, 0.9519927536231885, 0.9519927536231885, 0.9519927536231885, 0.9519927536231885,0.9519927536231885, .9519927536231885, 0.9519927536231885, 0.9519927536231885, 0.9519927536231885, 0.9519927536231885, 0.9519927536231885, 0.9519927536231885]
    ilpd = [0.5774509803921568, 0.5774509803921568, 0.5774509803921568, 0.5774509803921568, 0.5774509803921568, 0.5774509803921568, 0.5774509803921568, 0.5774509803921568, 0.5774509803921568, 0.5774509803921568, 0.5774509803921568, 0.5774509803921568, 0.5774509803921568, 0.5774509803921568, 0.5774509803921568, 0.5774509803921568, 0.5774509803921568, 0.5774509803921568, 0.5774509803921568, 0.5774509803921568]
    ttt = [0.7265625, 0.7265625, 0.7265625, 0.7265625, 0.7265625, 0.7265625, 0.7265625, 0.7265625, 0.7265625, 0.7265625, 0.7265625, 0.7265625, 0.7265625, 0.7265625, 0.7265625, 0.7265625, 0.7265625, 0.7265625, 0.7265625, 0.7265625]

    plt.figure()

    # for i in range(len(epsilons)):
    plt.plot(
        epsilons,
        ive,
        color=colors[0],
        lw=2,
        label="iris: iris-versicolor"
    )

    plt.plot(
        epsilons,
        ivi,
        color=colors[1],
        lw=2,
        label="iris: iris-virginica"
    )

    plt.plot(
        epsilons,
        ise,
        color=colors[2],
        lw=2,
        label="iris: iris-setosa"
    )

    plt.plot(
        epsilons,
        breast_w,
        color=colors[3],
        lw=2,
        label="breast-w"
    )

    plt.plot(
        epsilons,
        ilpd,
        color=colors[4],
        lw=2,
        label="ilpd"
    )

    plt.plot(
        epsilons,
        ttt,
        color=colors[5],
        lw=2,
        label="tic-tac-toe"
    )

    plt.xlim([0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Epsilon')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC al variare di epsilon')
    plt.legend(loc='lower right')
    plt.show()
