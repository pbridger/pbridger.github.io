import sys
from functools import reduce
from operator import add

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


index_col_types = {
    'torch-version': str, 'cuda-version': str, 'cudnn-version': str, 'model-arch': str,
    'optimize': str, 'batch-size': int, 'resolution-h': int, 'resolution-w': int, 'pixels': int,
    'iterations': int, 'inference-mode': str, 'no-grad': str, 'cudnn-benchmark': str,
    'autocast-dtype': str, 'model-cast-dtype': str, 'channels-last': str, 'pessimization': str, 'jit-script': str,
    'eval': str, 'compile': str, 'compile-mode': str, 'set-to-none': str, 'bnb': str, 'model-params': int,
    'mem-ckpt': str,
}

throughput, throughput_cost_pc, peak_memory_gb, memory_saving_pc = 'throughput (it/s)', 'throughput-cost (%)', 'peak-memory (GB)', 'memory-saving (%)'
metric_cols = [throughput, peak_memory_gb]

def index_val(col_names, row):
    return ',\n'.join([f'{c}={row[c]}' for c in col_names])

def group_by(cube, col_names):
    return [index_val(col_names, r) for ix, r in cube.iterrows()]

def col_names(prepend_col_names, remove_col_names):
    return prepend_col_names + list(set(index_col_types.keys()) - set(prepend_col_names) - set(remove_col_names))

def load_cube(filepath):
    cube = pd.read_csv(filepath, dtype={**index_col_types, **{c: float for c in metric_cols}})
    cube['peak-memory-label'] = cube[peak_memory_gb].map(lambda v: f'{v:.1f} GB')
    cube['peak-memory-label-2f'] = cube[peak_memory_gb].map(lambda v: f'{v:.2f} GB')

    cube['model-family'] = 'none'
    cube['model-family'][cube['model-arch'].str.contains(r'resnet')] = 'resnet'
    cube['model-family'][cube['model-arch'].str.contains(r'gpt')] = 'gpt'
    cube['model-family'][cube['model-arch'].str.contains(r'bert')] = 'bert'

    cube['torch-version'] = cube['torch-version'].map(lambda v: '.'.join(v.split('.')[:2]))

    cube['cast-dtype'] = [
        f'{c}-forced' if c != 'None' else f'{a}-auto'
        for a, c in zip(cube['autocast-dtype'], cube['model-cast-dtype'])
    ]

    cube['set-to-none'][cube['set-to-none'] == 'None'] = 'default'

#     cube['optimizer-text'] = [
#         ('Paged' if p else '') + o + (' (BnB)' if b == 'True' else '')
#         for o, b, p in zip(cube['optimizer'], cube['bnb'], cube['paged'])
#     ]

#     cube['optimizer-variant'] = [
#         ('BnB 8-bit' if b == 'True' else 'PyTorch MP') + (', Paged' if p else '')
#         for o, b, p in zip(cube['optimizer'], cube['bnb'], cube['paged'])
#     ]

    cube['mem-ckpt'] = cube['mem-ckpt'].map(lambda p: p.replace('\\', '').replace('$', ''))

    baseline_mask = cube['mem-ckpt'] == 'None'
    # baseline_mask = (cube['optimizer'] == 'Adam') & (cube['bnb'] == 'False')
    # baseline_mask = cube['set-to-none'] == 'False'

    baseline_gb_by_stage = {
        stage: cube[cube['stage'] == stage][baseline_mask][peak_memory_gb].max()
        for stage in cube['stage'].unique()
    }

    cube[memory_saving_pc] = [
        (100 * (baseline_gb_by_stage[stage] - peak_mem) / baseline_gb_by_stage[stage])
        for stage, peak_mem in zip(cube['stage'], cube[peak_memory_gb])
    ]

    baseline_throughput_by_stage = {
        stage: cube[cube['stage'] == stage][baseline_mask][throughput].max()
        for stage in cube['stage'].unique()
    }

    cube[throughput_cost_pc] = [
        (100 * (baseline_throughput_by_stage[stage] - tp) / baseline_throughput_by_stage[stage])
        for stage, tp in zip(cube['stage'], cube[throughput])
    ]

    print('loaded', filepath)
    print(cube)
    cube['dump'] = [index_val(index_col_types.keys(), row) for ix, row in cube.iterrows()]
    return cube


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
    fig.write_image(f'images/{n}.png', width=768, height=512, scale=2)
    print('wrote', n)

# sorted_colnames = col_names(['channels-last'], ['autocast-dtype'])

category_orders = {
    'stage': ['init', 'post-inputs', 'post-load', 'post-mode', 'post-mem-format', 'post-compile', 'post-opt-init', 'peak-warmup', 'post-warmup', 'peak-main', 'post-main'],
    'model-arch': ['resnet18', 'resnet50', 'bert-base-uncased', 'bert-large-uncased', 'distilgpt2'],
    'feature-set': ['worst', 'best'],
    'eval,inference-mode': ['False,False', 'False,True', 'True,False', 'True,True'],
    'cast-dtype': ['float32-auto', 'float16-auto', 'float16-forced'],
    'optimizer-variant': ['PyTorch MP', 'BnB 8-bit', 'BnB 8-bit, Paged'],
    'set-to-none': ['False', 'True', 'default'],
    **{b: ['False', 'True'] for b in ['optimize', 'inference-mode', 'no-grad', 'cudnn-benchmark', 'channels-last', 'pessimization', 'jit-script', 'eval', 'compile', 'bnb']}
}

# hover_data=index_col_types.keys(),

# cube = load_cube('set-to-none.csv')
# write_chart(
#     'set-to-none',
#     px.bar(
#         cube[cube['stage'] == 'peak-main-loop'],
#         x='torch-version',
#         y=peak_memory_gb,
#         text='peak-memory-label-2f',
#         color='set-to-none',
#         category_orders=category_orders,
#         barmode='group',
#     )#.update_traces(marker_size=16, textposition='middle right')
# )
# sys.exit()

#     px.scatter(
#         cube[cube['stage'] == 'peak-main-loop'],
#         x=memory_saving_pc,
#         y=throughput_cost_pc,
#         text='optimizer',
#         color='optimizer-variant',
#         category_orders=category_orders,
#     ).update_traces(marker_size=16, textposition='middle right')
# )

# cube = load_cube('inference-mode.csv')
# write_chart(
#     'components',
#     px.bar(
#         cube[cube['inference-mode'] == 'True'],
#         x='stage',
#         y='peak-memory (GB)',
#         text='peak-memory-label-2f',
#         category_orders=category_orders,
#         barmode='group',
#     )
# )

# cube = load_cube('mixed-precision.csv')
# write_chart(
#     'mixed-precision',
#     px.bar(
#         cube,
#         x='stage',
#         y='peak-memory (GB)',
#         text='peak-memory-label',
#         color='cast-dtype',
#         category_orders=category_orders,
#         barmode='group',
#     )
# )

# cube = load_cube('inference-mode.csv')
# write_chart(
#     'inference-mode',
#     px.bar(
#         cube,
#         x='stage',
#         y='peak-memory (GB)',
#         text='peak-memory-label',
#         color='inference-mode',
#         category_orders=category_orders,
#         barmode='group',
#     )
# )

# cube = load_cube('checkpointing-bert.csv')
# write_chart(
#     'checkpointing-bert',
#     px.scatter(
#         cube[cube['stage'] == 'peak-main-loop'],
#         x=peak_memory_gb,
#         y=throughput,
#         # text='peak-memory-label',
#         color='mem-ckpt',
#         category_orders=category_orders,
#     )#.update_yaxes(rangemode='tozero')
# )

# cube = load_cube('checkpointing-bert.csv')
# write_chart(
#     'checkpointing-bert',
#     px.scatter(
#         cube[cube['stage'] == 'peak-main-loop'],
#         x=memory_saving_pc,
#         y=throughput_cost_pc,
#         color='mem-ckpt',
#         category_orders=category_orders,
#     ).update_traces(marker_size=16)
# )

cube = load_cube('checkpointing-resnet.csv')
write_chart(
    'checkpointing-resnet',
    px.scatter(
        cube[cube['stage'] == 'peak-main-loop'],
        x=memory_saving_pc,
        y=throughput_cost_pc,
        color='mem-ckpt',
        category_orders=category_orders,
    ).update_traces(marker_size=16)
)

# cube = load_cube('optimizer-resnet.csv')
# write_chart(
#     'optimizer-resnet',
#     px.scatter(
#         cube[cube['stage'] == 'peak-main-loop'],
#         x=memory_saving_pc,
#         y=throughput_cost_pc,
#         text='optimizer',
#         color='optimizer-variant',
#         category_orders=category_orders,
#     ).update_traces(marker_size=16, textposition='middle right')
# )

# cube = load_cube('optimizer-bert.csv')
# write_chart(
#     'optimizer-bert',
#     px.scatter(
#         cube[cube['stage'] == 'peak-main-loop'],
#         x=memory_saving_pc,
#         y=throughput_cost_pc,
#         text='optimizer',
#         color='optimizer-variant',
#         category_orders=category_orders,
#     ).update_traces(marker_size=16, textposition='middle right')
# )






# print(cube[cube['inference-mode'] == 'False']['stage'])
# print(cube[cube['inference-mode'] == 'False']['delta-memory (GB)'])
# print()
# print(cube[cube['inference-mode'] == 'True']['stage'])
# print(cube[cube['inference-mode'] == 'True']['delta-memory (GB)'])

# fig = go.Figure()
# fig.add_trace(go.Waterfall(
#     x=cube[cube['inference-mode'] == 'False']['stage'],
#     y=cube[cube['inference-mode'] == 'False']['delta-memory (GB)'],
#     measure=['relative'] * 6,
#     increasing={'marker': {'color': 'rgb(99, 110, 250)', 'line': {'color': 'rgb(17, 17, 17)'}}},
#     connector={'line': {'color': 'rgb(238, 238, 238)'}, 'mode': 'between'},
# ))
# fig.add_trace(go.Waterfall(
#     x=cube[cube['inference-mode'] == 'True']['stage'],
#     y=cube[cube['inference-mode'] == 'True']['delta-memory (GB)'],
#     measure=['relative'] * 6,
#     increasing={'marker': {'color': 'rgb(239, 85, 59)', 'line': {'color': 'rgb(17, 17, 17)'}}},
#     connector={'line': {'color': 'rgb(238, 238, 238)'}, 'mode': 'between'},
# ))
#     # text=cube[cube['inference-mode'] == 'False']['peak-memory-label'],
#     # text=cube[cube['inference-mode'] == 'True']['peak-memory-label'],

# write_chart(
#     'inference-mode-w',
#     fig,
# )

import sys; sys.exit()


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


