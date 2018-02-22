import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import pandas as pd
import numpy as np

print (1 - norm.cdf((120-50) / 20))

def get_data():
    return pd.read_csv('~/Workspaces/MSC/statistics-and-data-analysis/ex2/UnivLyon_heart_disease_male.csv')

def multihist(data, labels, param):
    gmin = min(min(l) for l in data)
    gmax = max(max(l) for l in data)

    bins = range(int(gmin) - (int(gmin) % 10), int(gmax) + (int(gmax) % 10), 10)
    fig, ax = plt.subplots()
    weights = [np.ones_like(d) / len(d) for d in data]
    ax.hist(data, bins=bins, alpha=0.7, weights=weights, normed=False, label=labels)

    plt.xticks(bins)
    name = "max_heart_rate by {}".format(param)
    ax.set_title(name)
    ax.set_xlabel('Max Heart Rate')
    ax.set_ylabel('Frequency')

    plt.legend(loc=2, prop={'size': 12})
    plt.savefig(name.replace(' ', '_'))


df = get_data()

# params = [
#     'blood_sugar',
#     'chest_pain',
#     'exercice_angina',
#     'disease',
#     'rest_electro'
# ]
# for param in params:
#     labels = df[param].unique()
#     data = [df.where(df[param] == label).dropna()['max_heart_rate'] for label in labels]
#     multihist(data, list(labels), param)

# data = [
#     df.where(df['age'] > 50).dropna()['max_heart_rate'],
#     df.where(df['age'] <= 50).dropna()['max_heart_rate']
# ]

# multihist(data, ['over 50', 'under 50'], 'age')

# df['decade'] = df.age // 10
# ax = df.boxplot(column='max_heart_rate', by='decade')
# plt.title('')
# plt.ylabel('max heart rate')
# plt.savefig('box_plot_max_heart_rate_by_decade')


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x = df['age']
# y = df['max_heart_rate']
# z = df['rest_bpress']

# ax.scatter(x, y, z, c='r', marker='o')

# ax.set_xlabel('Age')
# ax.set_ylabel('Max Heart Rate')
# ax.set_zlabel('Rest Blood Press')

# plt.savefig('age_vs_max_heart_rate_vs_r_bpress')

# data = df['rest_bpress']
# mu, std = norm.fit(data)
# plt.hist(data, bins=10, normed=True, alpha=0.6)
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k', linewidth=2)
# title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
# plt.title(title)

# plt.savefig('norm_fit_r_bpress')


# ax = df.boxplot(column='max_heart_rate', by='blood_sugar')
# plt.title('')
# plt.ylabel('max heart rate')
# plt.savefig('box_plot_max_heart_rate_by_blood_sugar')

