from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')
Genotype_normal = namedtuple('Genotype_normal', 'normal normal_concat')
PRIMITIVES = [
    'none',
    'skip_connect',
    # 'conv_1x1',
    # 'edge_conv',
    # 'mr_conv',
    # 'gat',
    'gat',
    'gat_4',
    'gat_8',
    'gat_linear',
    'gat_cos',
    'gat_generalized_linear',
    'semi_gcn',
    'gin',
    'sage',
    'sage_sum',
    'sage_max',
    'appnp',
    'arma'
    # 'rel_sage',
]

Coauthor_CS = Genotype(normal=[('sage_sum', 0), ('gat_8', 1), ('gat_8', 1), ('gat_cos', 2), ('gat_4', 1), ('gat', 3), ('sage_mean', 1), ('gat_4', 4), ('arma', 0), ('semi_gcn', 1), ('sage_sum', 0), ('skip_connect', 5)], normal_concat=range(4, 8))
Coauthor_Physics = Genotype(normal=[('gat_8', 0), ('skip_connect', 1), ('gat_8', 0), ('sage_mean', 2), ('sage_mean', 1), ('gin', 2), ('skip_connect', 0), ('skip_connect', 4), ('arma', 3), ('gat_4', 5), ('skip_connect', 1), ('semi_gcn', 3)], normal_concat=range(4, 8))
Products_Best = Genotype(normal=[('skip_connect', 0), ('sage_mean', 1), ('gat_linear', 0), ('sage_mean', 2), ('gat_generalized_linear', 0), ('skip_connect', 3), ('gat', 0), ('gat_cos', 4), ('gin', 0), ('gat_cos', 1), ('gat', 0), ('gat_cos', 1), ('gat', 1), ('gat_linear', 7), ('gin', 1), ('gin', 7)], normal_concat=range(6, 10))
Arxiv_Best = Genotype(normal=[('sage_mean', 0), ('gat_cos', 1), ('gat_linear', 0), ('semi_gcn', 2), ('gat_cos', 0), ('appnp', 3), ('gat_8', 0), ('arma', 4), ('sage_mean', 1), ('gat_4', 5), ('gat', 0), ('arma', 6), ('sage_sum', 1), ('arma', 6)], normal_concat=range(5, 9))

Papers_Best = Genotype(normal=[('gat_generalized_linear', 0), ('gat_linear', 1), ('skip_connect', 0), ('gat_8', 1), ('gat', 0), ('semi_gcn', 1), ('sage_sum', 1), ('sage_sum', 4), ('gat_cos', 0), ('gat_4', 5), ('sage_mean', 1), ('gat_4', 6), ('sage_mean', 1), ('sage_sum', 7), ('semi_gcn', 0), ('gat_8', 7)], normal_concat=range(6, 10))