import sys
from functools import reduce
from operator import add, mul
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


col_types = {
    'model-arch': str,
    'model-params': int, 'model-flops-total': int, 'model-flops-conv': int, 'model-flops-batch_norm': int, 'model-flops-adaptive_avg_pool2d': int, 'model-flops-linear': int,
    '': int,
}

cube = pd.read_csv('layers.csv', dtype=col_types)
print(cube)

def write_chart(n, fig, legend={}):
    combined_legend = {
        'orientation': 'h',
        'yanchor': 'top',
        'y': 1.15,
        'xanchor': 'right',
        'x': 0.99,
        **legend
    }

    fig.update_traces(fill='toself')
    fig.update_layout(
        font_family='roboto, sans serif',
        margin={
            't': 12,
            'b': 12,
        },
        # legend=combined_legend
    )
    fig.write_html(f'html/{n}.html')
    fig.write_json(f'charts/{n}.json')
    print('wrote', n)

# sorted_colnames = col_names(['channels-last'], ['autocast-dtype'])

model_arches = ['resnet18', 'resnet50', 'bert-base-uncased', 'bert-large-uncased', 'distilgpt2']
flops_by_op = [c for c in cube.columns if c.startswith('model-flops-')]
layer_tally = [c for c in cube.columns if c.startswith('model-layers-')]

# print()
# print(cube['model-layers-layernorm'])
# print(cube['model-layers-batchnorm2d'])
# print(cube['model-layers-layernorm'] + cube['model-layers-batchnorm2d'])

cube['model-layers-total'] = reduce(add, [cube[c] for c in cube.columns if c.startswith('model-layers-')])
print(cube)

layer_tally = [
    'model-layers-relu',
    'model-layers-conv2d',
    'model-layers-conv1d',
    'model-layers-maxpool2d',
    'model-layers-adaptiveavgpool2d',
    'model-layers-linear',
    'model-layers-batchnorm2d',
    'model-layers-layernorm',
    'model-layers-dropout',
    'model-layers-embedding',
    'model-layers-newgeluactivation'
]

for m in model_arches:
    df = cube[cube['model-arch'] == m]

    write_chart(
        f'flops-by-op-{m}',
        px.line_polar(
            pd.DataFrame(dict(
                r=[df[c].iat[0] / df['model-flops-total'].iat[0] for c in flops_by_op],
                theta=[op[12:] for op in flops_by_op]
            )),
            r='r',
            theta='theta',
            line_close=True,
        )
    )

    write_chart(
        f'layer-tally-{m}',
        px.line_polar(
            pd.DataFrame(dict(
                r=[df[c].iat[0] for c in layer_tally],
                theta=[l[13:] for l in layer_tally],
            )),
            r='r',
            theta='theta',
            line_close=True,
        )
    )
    
