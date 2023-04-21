from functools import reduce
from operator import add

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


index_col_types = {
    'torch-version': str, 'cuda-version': str, 'cudnn-version': str, 'model-arch': str,
    'optimize': str, 'batch-size': int, 'resolution-h': int, 'resolution-w': int, 'pixels': int,
    'iterations': int, 'inference-mode': str, 'no-grad': str, 'cudnn-benchmark': str,
    'autocast-dtype': str, 'channels-last': str, 'pessimization': str, 'jit-script': str,
    'eval': str, 'compile': str, 'compile-mode': str, 'set-to-none': str, 'bnb': str, 'model-params': int,
    'model-flops-total': int, 'model-flops-conv': int, 'model-flops-batch_norm': int, 'model-flops-adaptive_avg_pool2d': int, 'model-flops-linear': int,
    'container-src': str
}

t = 'throughput (it/s)'
metric_cols = [t, 'peak-memory (GB)']
cube = pd.read_csv('cube.csv', dtype={**index_col_types, **{c: float for c in metric_cols}})

cube['model-family'] = 'none'
cube['model-family'][cube['model-arch'].str.contains(r'resnet')] = 'resnet'
cube['model-family'][cube['model-arch'].str.contains(r'gpt')] = 'gpt'
cube['model-family'][cube['model-arch'].str.contains(r'bert')] = 'bert'

cube['torch-version'] = cube['torch-version'].map(lambda v: '.'.join(v.split('.')[:2]))

cube[f'throughput (TFLOPS)'] = cube[t] * cube['model-flops-total'] / 1e12 / cube['batch-size']
cube[f'throughput (conv TFLOPS)'] = cube[t] * cube['model-flops-conv'] / 1e12 / cube['batch-size']
cube[f'throughput (linear TFLOPS)'] = cube[t] * cube['model-flops-linear'] / 1e12 / cube['batch-size']
print(cube)

def index_val(col_names, row):
    return ',\n'.join([f'{c}={row[c]}' for c in col_names])

def group_by(cube, col_names):
    return [index_val(col_names, r) for ix, r in cube.iterrows()]

def col_names(prepend_col_names, remove_col_names):
    return prepend_col_names + list(set(index_col_types.keys()) - set(prepend_col_names) - set(remove_col_names))

cube['dump'] = [index_val(index_col_types.keys(), row) for ix, row in cube.iterrows()]

def write_chart(n, fig, legend={}):
    combined_legend = {
        'orientation': 'h',
        'yanchor': 'top',
        'y': 1.10,
        'xanchor': 'right',
        'x': 0.99,
        **legend
    }

    fig.update_layout(
        font_family='roboto, sans serif',
        margin={
            't': 0,
            'b': 12,
        },
        legend=combined_legend,
        template='plotly_dark',
    )
    fig.write_html(f'html/{n}.html')
    fig.write_json(f'charts/{n}.json')
    print('wrote', n)

# sorted_colnames = col_names(['channels-last'], ['autocast-dtype'])

category_orders = {
    'model-arch': ['resnet18', 'resnet50', 'bert-base-uncased', 'bert-large-uncased', 'distilgpt2'],
    'feature-set': ['worst', 'best'],
    'eval,inference-mode': ['False,False', 'False,True', 'True,False', 'True,True'],
    **{b: ['False', 'True'] for b in ['optimize', 'inference-mode', 'no-grad', 'cudnn-benchmark', 'channels-last', 'pessimization', 'jit-script', 'eval', 'compile', 'set-to-none', 'bnb']}
}

# hover_data=index_col_types.keys(),

write_chart(
    'mixed-precision',
    px.scatter(
        cube,
        y='model-arch',
        x=f'throughput (TFLOPS)',
        color='autocast-dtype',
        # size='model-flops-total',
        category_orders=category_orders,
    )
)

write_chart(
    'channels-last',
    px.scatter(
        cube[
            (cube['autocast-dtype'] == 'float16') &
            (cube['compile'] == 'False') &
            (cube['eval'] == 'True') &
            (cube['model-family'] == 'resnet')
        ],
        y='model-arch',
        x=f'throughput (TFLOPS)',
        color='channels-last',
        # size='model-flops-total',
        category_orders=category_orders,
    ),
    legend={'y': 1.24}
)

write_chart(
    'channels-last-docker',
    px.scatter(
        cube[
            (cube['autocast-dtype'] == 'float16') &
            (cube['compile'] == 'False') &
            (cube['eval'] == 'True') &
            (cube['model-family'] == 'resnet')
        ],
        y='model-arch',
        x=f'throughput (TFLOPS)',
        color='channels-last',
        # size='model-flops-total',
        category_orders=category_orders,
        facet_row='container-src',
        # hover_data=index_col_types.keys(),
    ),
)

write_chart(
    'torch-compile',
    px.scatter(
        cube[
            ((cube['torch-version'] == '2.0') | (cube['torch-version'] == '2.1')) &
            (cube['autocast-dtype'] == 'float16') &
            (cube['channels-last'] == 'False')
        ],
        y='model-arch',
        x='throughput (TFLOPS)',
        color='compile',
        # size='model-flops-total',
        facet_row='torch-version',
        category_orders=category_orders,
    ),
    legend={'y': 1.07}
)

write_chart(
    'torch-compile-eval',
    px.scatter(
        cube[
            ((cube['torch-version'] == '2.0') | (cube['torch-version'] == '2.1')) &
            (cube['autocast-dtype'] == 'float16') &
            (cube['compile'] == 'True') &
            (cube['channels-last'] == 'False')
        ],
        y='model-arch',
        x=f'throughput (TFLOPS)',
        color='eval',
        # size='model-flops-total',
        facet_row='torch-version',
        category_orders=category_orders,
    ),
    legend={'y': 1.07}
)

write_chart(
    'cudnn-benchmark-channels-last',
    px.scatter(
        cube[
            (cube['autocast-dtype'] == 'float16') &
            (cube['eval'] == 'True') &
            (cube['compile'] == 'False') &
            (cube['container-src'] == 'ngc') &
            (cube['model-family'] == 'resnet')
        ],
        y='model-arch',
        x=f'throughput (TFLOPS)',
        color='cudnn-benchmark',
        # size='model-flops-total',
        facet_row='channels-last',
        category_orders=category_orders,
    )
)

write_chart(
    'cudnn-benchmark',
    px.scatter(
        cube[
            (cube['autocast-dtype'] == 'float16') &
            (cube['eval'] == 'True') &
            (cube['compile'] == 'False') &
            (cube['container-src'] == 'ngc') &
            (cube['channels-last'] == 'False')
        ],
        y='model-arch',
        x=f'throughput (TFLOPS)',
        color='cudnn-benchmark',
        # size='model-flops-total',
        category_orders=category_orders,
        # hover_data=index_col_types.keys(),
    ),
    # legend={'y': 1.11}
)

write_chart(
    'cudnn-benchmark-docker',
    px.scatter(
        cube[
            (cube['autocast-dtype'] == 'float16') &
            (cube['eval'] == 'True') &
            (cube['compile'] == 'False') &
            (cube['channels-last'] == 'False')
        ],
        y='model-arch',
        x=f'throughput (TFLOPS)',
        color='cudnn-benchmark',
        # size='model-flops-total',
        facet_row='container-src',
        category_orders=category_orders,
    ),
    legend={'y': 1.11}
)

write_chart(
    'inference-mode',
    px.scatter(
        cube[
            (cube['autocast-dtype'] == 'float16') &
            (cube['compile'] == 'False') &
            (cube['eval'] == 'True') &
            (cube['channels-last'] == 'False')
        ],
        y='model-arch',
        x=f'throughput (TFLOPS)',
        color='inference-mode',
        # size='model-flops-total',
        category_orders=category_orders,
    )
)

write_chart(
    'eval',
    px.scatter(
        cube[
            (cube['autocast-dtype'] == 'float16') &
            (cube['compile'] == 'False') &
            (cube['inference-mode'] == 'True') &
            (cube['channels-last'] == 'False')
        ],
        y='model-arch',
        x=f'throughput (TFLOPS)',
        color='eval',
        # size='model-flops-total',
        category_orders=category_orders,
    )
)

cube['eval,inference-mode'] = [f'{e},{i}' for e, i in zip(cube['eval'], cube['inference-mode'])]

write_chart(
    'eval-inference-mode',
    px.violin(
        cube[
            (cube['autocast-dtype'] == 'float16') &
            (cube['compile'] == 'False') &
            (cube['channels-last'] == 'False') #&
            # ((cube['eval-inference-mode'] == 'True-True') | (cube['eval-inference-mode'] == 'False-False'))
        ],
        y='model-arch',
        x=f'throughput (TFLOPS)',
        color='eval,inference-mode',
        # size='model-flops-total',
        category_orders=category_orders,
    )
)

write_chart(
    'torch-version',
    px.violin(
        cube[
            (cube['autocast-dtype'] == 'float16') &
            (cube['compile'] == 'False') &
            (cube['channels-last'] == 'False') &
            (cube['inference-mode'] == 'True')
        ],
        y='model-arch',
        x=f'throughput (TFLOPS)',
        color='torch-version',
        # size='model-flops-total',
        category_orders=category_orders,
    )
)

write_chart(
    'container-src',
    px.violin(
        cube[
            (cube['autocast-dtype'] == 'float16') &
            (cube['compile'] == 'False') &
            (cube['channels-last'] == 'False') &
            (cube['inference-mode'] == 'True')
        ],
        y='model-arch',
        x=f'throughput (TFLOPS)',
        color='container-src',
        # size='model-flops-total',
        category_orders=category_orders,
    )
)


worst = \
    (cube['autocast-dtype'] == 'float32') & \
    (cube['channels-last'] == 'False') & \
    (cube['inference-mode'] == 'False') & \
    (cube['cudnn-benchmark'] == 'False') & \
    (cube['eval'] == 'False')
    # (cube['compile'] == 'False') &

best = \
    (cube['autocast-dtype'] == 'float16') & \
    (((cube['model-family'] == 'resnet') & (cube['channels-last'] == 'True') & (cube['container-src'] == 'ngc')) |
     ((cube['model-family'] != 'resnet') & (cube['channels-last'] == 'False'))) & \
    (cube['inference-mode'] == 'True') & \
    (cube['cudnn-benchmark'] == 'True') & \
    (cube['eval'] == 'True')
    # (cube['compile'] == 'True') &

cube['feature-set'] = 'mixed'
cube['feature-set'][worst] = 'worst'
cube['feature-set'][best] = 'best'

write_chart(
    'feature-set',
    px.scatter(
        cube[worst | best],
        y='model-arch',
        x=f'throughput (TFLOPS)',
        color='feature-set',
        # size='model-flops-total',
        category_orders=category_orders,
    )
)


