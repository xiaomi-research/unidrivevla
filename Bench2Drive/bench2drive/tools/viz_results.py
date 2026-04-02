"""
Bench2Drive Evaluation Visualizer
Usage:
    cd /path/to/Bench2Drive
    streamlit run bench2drive/tools/viz_results.py

Dependencies: streamlit plotly pandas
"""

import os
import json
import glob
import re
import argparse
import sys

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_EVAL_ROOT = os.path.realpath(os.path.join(_SCRIPT_DIR, '../../evaluation'))

INFRACTION_COLS = [
    'collisions_layout', 'collisions_pedestrian', 'collisions_vehicle',
    'red_light', 'stop_infraction', 'outside_route_lanes',
    'min_speed_infractions', 'yield_emergency_vehicle_infractions',
    'scenario_timeouts', 'route_dev', 'vehicle_blocked', 'route_timeout',
]
INFRACTION_LABELS = {
    'collisions_layout':                   'Collision (layout)',
    'collisions_pedestrian':               'Collision (pedestrian)',
    'collisions_vehicle':                  'Collision (vehicle)',
    'red_light':                           'Red light',
    'stop_infraction':                     'Stop sign',
    'outside_route_lanes':                 'Off-lane',
    'min_speed_infractions':               'Min speed',
    'yield_emergency_vehicle_infractions': 'Yield emergency',
    'scenario_timeouts':                   'Scenario timeout',
    'route_dev':                           'Route deviation',
    'vehicle_blocked':                     'Blocked',
    'route_timeout':                       'Route timeout',
}
PENALTY_COLS = [c for c in INFRACTION_COLS if c != 'min_speed_infractions']

STATUS_COLOR = {
    'Perfect':   '#2ecc71',
    'Completed': '#3498db',
    'Failed':    '#e74c3c',
    'other':     '#95a1a6',
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _route_num(rid):
    """Extract integer from 'RouteScenario_4937_rep0' → 4937."""
    m = re.search(r'(\d+)', rid)
    return int(m.group(1)) if m else 0


# cache_data introduced in 1.18; fall back to legacy st.cache for older installs
if hasattr(st, 'cache_data'):
    _cache = st.cache_data
else:
    def _cache(func=None, **kwargs):          # bare @_cache or @_cache(...)
        import functools
        decorator = st.cache(allow_output_mutation=True, suppress_st_warning=True)
        if func is not None:
            return decorator(func)
        def wrapper(f):
            return decorator(f)
        return wrapper


@_cache(show_spinner='Loading experiments...')
def load_all_experiments(eval_root: str):
    """Return (summary_df, routes_df)."""
    subdirs = sorted([
        d for d in os.listdir(eval_root)
        if os.path.isdir(os.path.join(eval_root, d))
    ])

    summary_rows = []
    route_rows = []

    for name in subdirs:
        merged = os.path.join(eval_root, name, 'merged.json')
        if not os.path.exists(merged):
            continue
        try:
            with open(merged) as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue

        n = data.get('eval num', 0)
        if n == 0:
            continue

        summary_rows.append({
            'experiment': name,
            'driving_score': data.get('driving score', 0.0),
            'success_rate':  data.get('success rate', 0.0) * 100,
            'eval_num':      n,
        })

        for rd in data['_checkpoint']['records']:
            row = {
                'experiment':    name,
                'route_id':      rd['route_id'],
                'route_num':     _route_num(rd['route_id']),
                'scenario_name': rd.get('scenario_name', ''),
                'weather_id':    rd.get('weather_id', ''),
                'town_name':     rd.get('town_name', ''),
                'status':        rd.get('status', ''),
                'score_route':   rd['scores']['score_route'],
                'score_penalty': rd['scores']['score_penalty'],
                'score_composed':rd['scores']['score_composed'],
                'success':       _is_success(rd),
            }
            for col in INFRACTION_COLS:
                row[col] = len(rd['infractions'].get(col, []))
            route_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values('driving_score', ascending=False)
    routes_df  = pd.DataFrame(route_rows)
    return summary_df, routes_df


def _is_success(rd):
    if rd['status'] not in ('Completed', 'Perfect'):
        return False
    for k, v in rd['infractions'].items():
        if len(v) > 0 and k != 'min_speed_infractions':
            return False
    return True


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------
def _status_color_map(statuses):
    return [STATUS_COLOR.get(s, STATUS_COLOR['other']) for s in statuses]


def page_overview(summary_df, routes_df):
    st.header('Overview — all experiments')

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    best = summary_df.iloc[0]
    c1.metric('Experiments', len(summary_df))
    c2.metric('Best Driving Score', f"{best['driving_score']:.2f}", best['experiment'][:30])
    c3.metric('Best Success Rate', f"{best['success_rate']:.1f}%")
    c4.metric('Max Routes', int(summary_df['eval_num'].max()))

    st.markdown("---")

    # Driving score bar
    fig = px.bar(
        summary_df.sort_values('driving_score'),
        x='driving_score', y='experiment',
        orientation='h',
        color='driving_score',
        color_continuous_scale='RdYlGn',
        range_color=[summary_df['driving_score'].min() * 0.95,
                     summary_df['driving_score'].max() * 1.02],
        hover_data=['success_rate', 'eval_num'],
        title='Driving Score (all experiments)',
        labels={'driving_score': 'Driving Score', 'experiment': ''},
        height=max(350, len(summary_df) * 28),
    )
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=0))
    st.plotly_chart(fig, use_container_width=True)

    # Success rate bar (side by side)
    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(
            summary_df.sort_values('success_rate'),
            x='success_rate', y='experiment', orientation='h',
            color='success_rate', color_continuous_scale='Blues',
            title='Success Rate (%)',
            labels={'success_rate': 'Success Rate (%)', 'experiment': ''},
            height=max(350, len(summary_df) * 28),
        )
        fig2.update_layout(coloraxis_showscale=False, margin=dict(l=0))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.scatter(
            summary_df,
            x='driving_score', y='success_rate',
            text='experiment',
            title='Driving Score vs Success Rate',
            labels={'driving_score': 'Driving Score', 'success_rate': 'Success Rate (%)'},
            height=max(350, len(summary_df) * 28),
        )
        fig3.update_traces(textposition='top center', textfont_size=9)
        st.plotly_chart(fig3, use_container_width=True)

    # Summary table
    st.subheader('Summary table')
    disp = summary_df.copy()
    disp['driving_score'] = disp['driving_score'].map('{:.4f}'.format)
    disp['success_rate']  = disp['success_rate'].map('{:.2f}%'.format)
    st.dataframe(disp.reset_index(drop=True), use_container_width=True)


def page_detail(summary_df, routes_df):
    st.header('Detail — single experiment')

    exp = st.selectbox('Select experiment', summary_df['experiment'].tolist())
    df = routes_df[routes_df['experiment'] == exp].copy()

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Routes', len(df))
    c2.metric('Driving Score', f"{df['score_composed'].mean():.2f}")
    c3.metric('Success Rate', f"{df['success'].mean()*100:.1f}%")
    c4.metric('Avg Route Score', f"{df['score_route'].mean():.1f}")

    st.markdown("---")

    # Route scatter — all 220 routes
    st.subheader('Per-route scores')
    df_sorted = df.sort_values('route_num')

    status_order = ['Perfect', 'Completed', 'Failed'] + [
        s for s in df['status'].unique() if s not in ('Perfect', 'Completed', 'Failed')
    ]
    color_map = {k: v for k, v in STATUS_COLOR.items()}

    fig = px.scatter(
        df_sorted, x='route_num', y='score_composed',
        color='status', color_discrete_map=color_map,
        category_orders={'status': status_order},
        hover_data=['route_id', 'scenario_name', 'town_name',
                    'score_route', 'score_penalty'],
        title=f'Score per route — {exp}',
        labels={'route_num': 'Route Number', 'score_composed': 'Composed Score'},
        height=420,
    )
    fig.add_hline(y=df['score_composed'].mean(), line_dash='dash',
                  annotation_text=f'mean={df["score_composed"].mean():.2f}',
                  line_color='gray')
    st.plotly_chart(fig, use_container_width=True)

    # Infraction breakdown
    st.subheader('Infraction breakdown')
    col1, col2 = st.columns(2)
    with col1:
        inf_counts = {INFRACTION_LABELS[c]: df[c].sum()
                      for c in INFRACTION_COLS if df[c].sum() > 0}
        if inf_counts:
            fig_inf = px.bar(
                x=list(inf_counts.values()), y=list(inf_counts.keys()),
                orientation='h', title='Total infractions (count)',
                labels={'x': 'Count', 'y': ''},
                color=list(inf_counts.values()),
                color_continuous_scale='Reds',
                height=350,
            )
            fig_inf.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_inf, use_container_width=True)
        else:
            st.success('No infractions recorded!')

    with col2:
        # Routes affected (how many routes had each infraction type)
        affected = {INFRACTION_LABELS[c]: (df[c] > 0).sum()
                    for c in INFRACTION_COLS if (df[c] > 0).sum() > 0}
        if affected:
            fig_aff = px.pie(
                names=list(affected.keys()), values=list(affected.values()),
                title='Routes affected per infraction type',
                height=350,
            )
            st.plotly_chart(fig_aff, use_container_width=True)

    # Score distribution
    st.subheader('Score distribution')
    col3, col4 = st.columns(2)
    with col3:
        fig_hist = px.histogram(
            df, x='score_composed', nbins=20,
            title='Composed score distribution',
            labels={'score_composed': 'Score'},
            color_discrete_sequence=['#3498db'],
            height=300,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    with col4:
        fig_box = px.box(
            df, y='score_composed', color='status',
            color_discrete_map=color_map,
            title='Score by status',
            height=300,
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Town breakdown
    st.subheader('Town / weather breakdown')
    col5, col6 = st.columns(2)
    with col5:
        town_stats = df.groupby('town_name').agg(
            routes=('route_id', 'count'),
            avg_score=('score_composed', 'mean'),
            success=('success', 'mean'),
        ).reset_index()
        town_stats['success'] = town_stats['success'] * 100
        fig_town = px.bar(
            town_stats.sort_values('avg_score'),
            x='avg_score', y='town_name', orientation='h',
            color='avg_score', color_continuous_scale='RdYlGn',
            hover_data=['routes', 'success'],
            title='Avg score by town',
            labels={'avg_score': 'Avg Score', 'town_name': ''},
            height=max(300, len(town_stats) * 30),
        )
        fig_town.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_town, use_container_width=True)
    with col6:
        scenario_stats = df.groupby('scenario_name').agg(
            routes=('route_id', 'count'),
            avg_score=('score_composed', 'mean'),
            success=('success', 'mean'),
        ).reset_index().sort_values('avg_score')
        scenario_stats['success'] = scenario_stats['success'] * 100
        fig_sc = px.bar(
            scenario_stats,
            x='avg_score', y='scenario_name', orientation='h',
            color='avg_score', color_continuous_scale='RdYlGn',
            hover_data=['routes', 'success'],
            title='Avg score by scenario',
            labels={'avg_score': 'Avg Score', 'scenario_name': ''},
            height=max(300, len(scenario_stats) * 22),
        )
        fig_sc.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_sc, use_container_width=True)

    # Route detail table
    st.subheader('Route detail table')
    filter_status = st.multiselect(
        'Filter by status', options=df['status'].unique().tolist(),
        default=df['status'].unique().tolist(), key='detail_status_filter'
    )
    show_cols = ['route_id', 'town_name', 'scenario_name', 'status',
                 'score_composed', 'score_route', 'score_penalty'] + \
                [c for c in INFRACTION_COLS if df[c].sum() > 0]
    tbl = df[df['status'].isin(filter_status)][show_cols].sort_values(
        'score_composed').reset_index(drop=True)
    tbl.columns = [INFRACTION_LABELS.get(c, c) for c in tbl.columns]
    st.dataframe(tbl, use_container_width=True, height=400)


def page_compare(summary_df, routes_df):
    st.header('Compare — multiple experiments')

    all_exps = summary_df['experiment'].tolist()
    selected = st.multiselect(
        'Select experiments to compare (2 or more)',
        options=all_exps,
        default=all_exps[:min(4, len(all_exps))],
    )
    if len(selected) < 2:
        st.warning('Select at least 2 experiments.')
        return

    df_sel = routes_df[routes_df['experiment'].isin(selected)]

    # ── Aggregate comparison bar ──────────────────────────────────────────
    st.subheader('Aggregate metrics comparison')
    agg = df_sel.groupby('experiment').agg(
        driving_score=('score_composed', 'mean'),
        success_rate=('success', 'mean'),
        avg_route_score=('score_route', 'mean'),
        avg_penalty=('score_penalty', 'mean'),
    ).reset_index()
    agg['success_rate'] *= 100
    agg = agg.set_index('experiment').loc[selected].reset_index()

    metrics = ['driving_score', 'success_rate', 'avg_route_score', 'avg_penalty']
    metric_labels = {
        'driving_score':    'Driving Score',
        'success_rate':     'Success Rate (%)',
        'avg_route_score':  'Avg Route Score',
        'avg_penalty':      'Avg Penalty',
    }
    fig_agg = go.Figure()
    for m in metrics:
        fig_agg.add_trace(go.Bar(
            name=metric_labels[m],
            x=agg['experiment'],
            y=agg[m],
            text=agg[m].map('{:.2f}'.format),
            textposition='outside',
        ))
    fig_agg.update_layout(
        barmode='group', title='Aggregate metrics comparison',
        xaxis_tickangle=-30, height=420,
        legend=dict(orientation='h', y=1.1),
    )
    st.plotly_chart(fig_agg, use_container_width=True)

    # ── Radar chart ────────────────────────────────────────────────────────
    st.subheader('Radar comparison')
    # Normalize each metric to 0–1 across selected experiments
    radar_metrics = list(metric_labels.keys())
    norm = agg.copy()
    for m in radar_metrics:
        mn, mx = norm[m].min(), norm[m].max()
        norm[m] = (norm[m] - mn) / (mx - mn + 1e-9)

    fig_radar = go.Figure()
    for _, row in norm.iterrows():
        vals = [row[m] for m in radar_metrics]
        vals += vals[:1]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=[metric_labels[m] for m in radar_metrics] + [metric_labels[radar_metrics[0]]],
            fill='toself', name=row['experiment'],
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title='Normalized metric radar (higher=better)',
        height=420,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Per-route heatmap ─────────────────────────────────────────────────
    st.subheader('Per-route score heatmap')
    pivot = df_sel.pivot_table(
        index='route_num', columns='experiment',
        values='score_composed', aggfunc='first'
    )[selected]

    # Sort routes by mean score
    pivot['_mean'] = pivot.mean(axis=1)
    sort_opt = st.radio('Sort routes by', ['route number', 'mean score (asc)', 'mean score (desc)'])
    if sort_opt == 'mean score (asc)':
        pivot = pivot.sort_values('_mean')
    elif sort_opt == 'mean score (desc)':
        pivot = pivot.sort_values('_mean', ascending=False)
    else:
        pivot = pivot.sort_index()
    pivot = pivot.drop(columns=['_mean'])

    fig_heat = px.imshow(
        pivot.T,
        color_continuous_scale='RdYlGn',
        zmin=0, zmax=100,
        aspect='auto',
        title='Score heatmap: experiments × routes',
        labels={'x': 'Route Number', 'y': 'Experiment', 'color': 'Score'},
        height=max(300, len(selected) * 50 + 100),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Delta analysis (pick 2) ────────────────────────────────────────────
    st.subheader('Delta analysis (pick any 2)')
    col1, col2 = st.columns(2)
    with col1:
        exp_a = st.selectbox('Experiment A', selected, index=0, key='delta_a')
    with col2:
        exp_b = st.selectbox('Experiment B', selected,
                             index=min(1, len(selected)-1), key='delta_b')

    if exp_a != exp_b:
        da = routes_df[routes_df['experiment'] == exp_a][['route_num', 'route_id', 'score_composed', 'status']].copy()
        db = routes_df[routes_df['experiment'] == exp_b][['route_num', 'route_id', 'score_composed', 'status']].copy()
        merged_ab = pd.merge(da, db, on=['route_num', 'route_id'], suffixes=('_A', '_B'))
        merged_ab['delta'] = merged_ab['score_composed_B'] - merged_ab['score_composed_A']
        merged_ab = merged_ab.sort_values('delta')

        fig_delta = px.bar(
            merged_ab, x='route_num', y='delta',
            color='delta',
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0,
            hover_data=['route_id', 'score_composed_A', 'score_composed_B', 'status_A', 'status_B'],
            title=f'Score delta: {exp_b} − {exp_a}  (positive = B better)',
            labels={'route_num': 'Route Number', 'delta': f'Δ Score (B−A)'},
            height=380,
        )
        fig_delta.add_hline(y=0, line_color='black', line_width=1)
        st.plotly_chart(fig_delta, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric('A avg', f"{merged_ab['score_composed_A'].mean():.2f}", exp_a[:25])
        c2.metric('B avg', f"{merged_ab['score_composed_B'].mean():.2f}", exp_b[:25])
        delta_mean = merged_ab['delta'].mean()
        c3.metric('Mean Δ (B−A)', f"{delta_mean:+.2f}")

    # ── Intersection analysis ─────────────────────────────────────────────
    st.subheader('Intersection analysis')
    st.caption('Classify each route by how many selected experiments pass/fail it.')

    # For each route: count how many exps pass
    success_pivot = df_sel.pivot_table(
        index='route_num', columns='experiment',
        values='success', aggfunc='first'
    )[selected].fillna(0)

    success_pivot['pass_count'] = success_pivot.sum(axis=1).astype(int)
    success_pivot['fail_count'] = len(selected) - success_pivot['pass_count']

    n = len(selected)
    counts = {
        f'All {n} pass':   (success_pivot['pass_count'] == n).sum(),
        f'All {n} fail':   (success_pivot['fail_count'] == n).sum(),
        f'Mixed':          ((success_pivot['pass_count'] > 0) & (success_pivot['fail_count'] > 0)).sum(),
    }
    col1, col2, col3 = st.columns(3)
    col1.metric(f'All {n} pass', int(counts[f'All {n} pass']))
    col2.metric(f'Mixed', int(counts['Mixed']))
    col3.metric(f'All {n} fail', int(counts[f'All {n} fail']))

    fig_inter = px.histogram(
        success_pivot.reset_index(),
        x='pass_count',
        nbins=n+1,
        title='Distribution: # experiments that pass each route',
        labels={'pass_count': '# experiments passing', 'count': 'Routes'},
        color_discrete_sequence=['#3498db'],
        height=300,
    )
    st.plotly_chart(fig_inter, use_container_width=True)

    # Routes all exps fail
    all_fail_routes = success_pivot[success_pivot['pass_count'] == 0].index.tolist()
    if all_fail_routes:
        with st.expander(f'Routes ALL experiments fail ({len(all_fail_routes)} routes)'):
            fail_df = routes_df[
                (routes_df['route_num'].isin(all_fail_routes)) &
                (routes_df['experiment'] == selected[0])
            ][['route_num', 'route_id', 'town_name', 'scenario_name', 'score_composed', 'status']].sort_values('route_num')
            st.dataframe(fail_df.reset_index(drop=True), use_container_width=True)

    # Routes all exps pass
    all_pass_routes = success_pivot[success_pivot['pass_count'] == n].index.tolist()
    if all_pass_routes:
        with st.expander(f'Routes ALL experiments pass ({len(all_pass_routes)} routes)'):
            pass_df = routes_df[
                (routes_df['route_num'].isin(all_pass_routes)) &
                (routes_df['experiment'] == selected[0])
            ][['route_num', 'route_id', 'town_name', 'scenario_name', 'score_composed', 'status']].sort_values('route_num')
            st.dataframe(pass_df.reset_index(drop=True), use_container_width=True)

    # Infraction comparison
    st.subheader('Infraction type comparison')
    inf_comp = df_sel.groupby('experiment')[INFRACTION_COLS].sum().reset_index()
    inf_melted = inf_comp.melt(id_vars='experiment', var_name='infraction', value_name='count')
    inf_melted['infraction'] = inf_melted['infraction'].map(INFRACTION_LABELS)
    inf_melted = inf_melted[inf_melted['count'] > 0]
    if not inf_melted.empty:
        fig_inf_cmp = px.bar(
            inf_melted, x='infraction', y='count', color='experiment',
            barmode='group',
            title='Infraction counts by type across experiments',
            labels={'infraction': 'Infraction Type', 'count': 'Count'},
            height=400,
        )
        fig_inf_cmp.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_inf_cmp, use_container_width=True)


def page_ranking(summary_df, routes_df):
    st.header('Ranking')

    rank_tab = st.radio(
        'Section',
        ['Experiment ranking', 'Route ranking', 'Worst routes'],
        key='rank_tab',
    )
    st.markdown("---")

    # ── Section 1: Experiment ranking ────────────────────────────────────
    if rank_tab == 'Experiment ranking':
        st.subheader('Experiment ranking')
        sort_by = st.radio('Sort by', ['driving_score', 'success_rate', 'eval_num'],
                           format_func=lambda x: x.replace('_', ' ').title(),
                           key='exp_sort')
        ranked = summary_df.sort_values(sort_by, ascending=False).reset_index(drop=True)
        ranked.index = ranked.index + 1
        ranked.index.name = 'Rank'
        ranked['driving_score'] = ranked['driving_score'].map('{:.4f}'.format)
        ranked['success_rate']  = ranked['success_rate'].map('{:.2f}%'.format)
        st.dataframe(ranked, use_container_width=True)

    # ── Section 2: Route ranking within one experiment ────────────────────
    elif rank_tab == 'Route ranking':
        st.subheader('Route ranking — single experiment')
        exp = st.selectbox('Experiment', summary_df['experiment'].tolist(), key='rank_exp')
        df = routes_df[routes_df['experiment'] == exp].copy()

        sort_col = st.radio('Sort by', ['score_composed', 'score_route', 'score_penalty'],
                            key='route_sort')
        ascending = st.checkbox('Ascending (worst first)', value=True, key='route_asc')

        ranked_routes = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
        ranked_routes.index = ranked_routes.index + 1
        ranked_routes.index.name = 'Rank'

        show = ['route_id', 'town_name', 'scenario_name', 'status',
                'score_composed', 'score_route', 'score_penalty'] + \
               [c for c in INFRACTION_COLS if df[c].sum() > 0]
        disp = ranked_routes[show].copy()
        disp.columns = [INFRACTION_LABELS.get(c, c) for c in disp.columns]
        st.dataframe(disp, use_container_width=True, height=500)

    # ── Section 3: Worst routes across experiments ────────────────────────
    elif rank_tab == 'Worst routes':
        st.subheader('Worst routes across all experiments')
        all_exps = summary_df['experiment'].tolist()
        exps_for_worst = st.multiselect(
            'Include experiments', all_exps,
            default=all_exps, key='worst_exps'
        )
        if not exps_for_worst:
            st.warning('Select at least one experiment.')
            return

        top_n = st.slider('Show worst N routes', 10, 100, 30)

        df_w = routes_df[routes_df['experiment'].isin(exps_for_worst)]
        worst = df_w.groupby('route_num').agg(
            route_id=('route_id', 'first'),
            avg_score=('score_composed', 'mean'),
            min_score=('score_composed', 'min'),
            max_score=('score_composed', 'max'),
            pass_rate=('success', 'mean'),
            town=('town_name', 'first'),
            scenario=('scenario_name', 'first'),
            appearances=('experiment', 'count'),
        ).reset_index().sort_values('avg_score').head(top_n)
        worst['pass_rate'] = (worst['pass_rate'] * 100).map('{:.1f}%'.format)
        worst['avg_score'] = worst['avg_score'].map('{:.2f}'.format)
        worst['min_score'] = worst['min_score'].map('{:.2f}'.format)
        worst['max_score'] = worst['max_score'].map('{:.2f}'.format)
        worst.index = range(1, len(worst) + 1)
        worst.index.name = 'Rank'
        st.dataframe(worst.drop(columns=['route_num']), use_container_width=True, height=500)

        # Heatmap of worst routes across experiments
        df_heat = df_w[df_w['route_num'].isin(worst['route_num'].unique())]
        pivot = df_heat.pivot_table(
            index='route_num', columns='experiment',
            values='score_composed', aggfunc='first'
        )[exps_for_worst].sort_values(exps_for_worst[0])

        fig_h = px.imshow(
            pivot.T,
            color_continuous_scale='RdYlGn',
            zmin=0, zmax=100,
            aspect='auto',
            title=f'Worst {top_n} routes — score heatmap',
            labels={'x': 'Route Number', 'y': 'Experiment', 'color': 'Score'},
            height=max(300, len(exps_for_worst) * 50 + 100),
        )
        st.plotly_chart(fig_h, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title='Bench2Drive Results Visualizer',
        layout='wide',
        initial_sidebar_state='expanded',
    )

    # Sidebar
    with st.sidebar:
        st.title('B2D Visualizer')
        eval_root = st.text_input(
            'Evaluation root', value=_DEFAULT_EVAL_ROOT
        )
        if st.button('Reload data'):
            try:
                st.cache_data.clear()
            except AttributeError:
                st.experimental_memo.clear()
            except Exception:
                pass
        st.markdown("---")
        st.caption(f'Root: {eval_root}')

    if not os.path.isdir(eval_root):
        st.error(f'Directory not found: {eval_root}')
        st.stop()

    summary_df, routes_df = load_all_experiments(eval_root)

    if summary_df.empty:
        st.warning('No merged.json files found. Run merge_route_json.py first.')
        st.stop()

    with st.sidebar:
        st.success(f'{len(summary_df)} experiments loaded')
        st.caption(f'{len(routes_df)} total route records')

    # Navigation
    page = st.sidebar.radio(
        'Page',
        ['Overview', 'Detail', 'Compare', 'Ranking'],
        index=0,
    )

    if page == 'Overview':
        page_overview(summary_df, routes_df)
    elif page == 'Detail':
        page_detail(summary_df, routes_df)
    elif page == 'Compare':
        page_compare(summary_df, routes_df)
    elif page == 'Ranking':
        page_ranking(summary_df, routes_df)


if __name__ == '__main__':
    main()
