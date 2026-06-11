import numpy as np
import pandas as pd
import yfinance as yf
from scipy.spatial.distance import pdist, squareform, jensenshannon
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def download_data():
    print("Downloading EURUSD data...")
    try:
        df = yf.download("EURUSD=X", start="2003-12-01")
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if 'Close' in df.columns:
                df = df[['Close']].copy()
            else:
                raise ValueError("Close column missing")
        else:
            raise ValueError("Empty dataframe from yfinance")
    except Exception as e:
        print(f"yfinance failed ({e}), generating mock data for testing...")
        dates = pd.date_range(start="2003-12-01", periods=5000, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0, 0.005, 5000)
        prices = 1.2 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({'Close': prices}, index=dates)
        
    df.dropna(inplace=True)
    return df

def prepare_data(df):
    print("Preparing data...")
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)
    q30 = df['log_ret'].quantile(0.3)
    q70 = df['log_ret'].quantile(0.7)
    conditions = [
        (df['log_ret'] < q30),
        (df['log_ret'] >= q30) & (df['log_ret'] <= q70),
        (df['log_ret'] > q70)
    ]
    choices = [-1, 0, 1]
    df['State'] = np.select(conditions, choices, default=np.nan)
    return df

def construct_states(df, lookback=5):
    print(f"Constructing states for M{lookback}...")
    for i in range(1, lookback + 1):
        df[f's{lookback - i + 1}'] = df['State'].shift(i)
    df['Y'] = df['State']
    df.dropna(inplace=True)
    state_cols = [f's{i}' for i in range(1, lookback + 1)]
    # Use list comprehension to create tuple of states to avoid apply return type issues on empty dataframes
    df['State_Tuple'] = [tuple(x) for x in df[state_cols].astype(int).values]
    return df, state_cols

def compute_transitions(df):
    print("Computing transition tables and Bayesian shrinkage...")
    transition_counts = df.groupby(['State_Tuple', 'Y']).size().unstack(fill_value=0)
    for target in [-1, 0, 1]:
        if target not in transition_counts.columns:
            transition_counts[target] = 0
    transition_counts = transition_counts[[-1, 0, 1]]
    state_counts = transition_counts.sum(axis=1)
    state_probability = state_counts / state_counts.sum()
    
    alpha = 5
    global_counts = transition_counts.sum(axis=0)
    global_probs = global_counts / global_counts.sum()
    
    transition_probs_smoothed = (transition_counts + alpha * global_probs).div(state_counts + alpha, axis=0)
    
    direction = transition_probs_smoothed[1] - transition_probs_smoothed[-1]
    eps = 1e-15
    entropy = -np.sum(transition_probs_smoothed * np.log(transition_probs_smoothed + eps), axis=1)
    gini = 1 - np.sum(transition_probs_smoothed**2, axis=1)
    confidence = transition_probs_smoothed.max(axis=1)
    classification_error = 1 - confidence
    
    states_df = pd.DataFrame({
        'State_Count': state_counts,
        'State_Probability': state_probability,
        'P(-1)': transition_probs_smoothed[-1],
        'P(0)': transition_probs_smoothed[0],
        'P(+1)': transition_probs_smoothed[1],
        'Direction': direction,
        'Entropy': entropy,
        'Gini': gini,
        'Confidence': confidence,
        'Classification_Error': classification_error,
        'Raw_Count(-1)': transition_counts[-1],
        'Raw_Count(0)': transition_counts[0],
        'Raw_Count(1)': transition_counts[1]
    })
    return states_df

def cluster_js(states_df):
    print("Clustering Method 1: Jensen-Shannon...")
    probs = states_df[['P(-1)', 'P(0)', 'P(+1)']].values
    
    n = len(probs)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = jensenshannon(probs[i], probs[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            
    condensed_dist = squareform(dist_matrix)
    Z = linkage(condensed_dist, method='average')
    
    best_k = 2
    best_score = -1
    best_labels = None
    for k in range(2, min(21, n)):
        labels = fcluster(Z, k, criterion='maxclust')
        if len(np.unique(labels)) > 1:
            score = silhouette_score(dist_matrix, labels, metric='precomputed')
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                
    if best_labels is None:
        best_labels = fcluster(Z, 2, criterion='maxclust')
        best_k = 2
        best_score = 0
                
    return best_labels, best_k, best_score, Z

def cluster_sse(states_df):
    print("Clustering Method 2: SSE / Euclidean...")
    probs = states_df[['P(-1)', 'P(0)', 'P(+1)']].values
    dist_matrix = pdist(probs, metric='euclidean')
    Z = linkage(dist_matrix, method='average')
    
    best_k = 2
    best_score = -1
    best_labels = None
    n = len(probs)
    for k in range(2, min(21, n)):
        labels = fcluster(Z, k, criterion='maxclust')
        if len(np.unique(labels)) > 1:
            score = silhouette_score(probs, labels, metric='euclidean')
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                
    if best_labels is None:
        best_labels = fcluster(Z, 2, criterion='maxclust')
        best_k = 2
        best_score = 0
                
    return best_labels, best_k, best_score, Z

def tree_clustering(states_df, df, criterion):
    print(f"Clustering Method: Tree ({criterion})...")
    states_df['Dominant'] = states_df[['P(-1)', 'P(0)', 'P(+1)']].idxmax(axis=1)
    
    feature_col = 'Entropy' if criterion == 'entropy' else 'Gini'
    X = states_df[['Direction', feature_col]].values
    
    map_target = {'P(-1)': -1, 'P(0)': 0, 'P(+1)': 1}
    y = states_df['Dominant'].map(map_target).values
    
    best_depth = 1
    best_score = -1
    for depth in range(1, 11):
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=42)
        if len(np.unique(y)) > 1:
            n_splits = min(5, np.min(np.unique(y, return_counts=True)[1]))
            if n_splits > 1:
                try:
                    scores = cross_val_score(clf, X, y, cv=n_splits)
                    score = scores.mean()
                    if score > best_score:
                        best_score = score
                        best_depth = depth
                except Exception as e:
                    pass
    
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=best_depth, random_state=42)
    clf.fit(X, y)
    labels = clf.apply(X)
    return labels, best_depth, best_score, clf

def reestimate_clusters(states_df, labels):
    states_df['Cluster'] = labels
    pooled = states_df.groupby('Cluster')[['Raw_Count(-1)', 'Raw_Count(0)', 'Raw_Count(1)']].sum()
    
    row_sums = pooled.sum(axis=1)
    probs = pooled.div(row_sums, axis=0)
    
    direction = probs['Raw_Count(1)'] - probs['Raw_Count(-1)']
    
    eps = 1e-15
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    gini = 1 - np.sum(probs**2, axis=1)
    confidence = probs.max(axis=1)
    class_error = 1 - confidence
    
    cluster_summary = pd.DataFrame({
        'Cluster_ID': pooled.index,
        'N_States': states_df.groupby('Cluster').size(),
        'N_Observations': row_sums,
        'P(-1)': probs['Raw_Count(-1)'],
        'P(0)': probs['Raw_Count(0)'],
        'P(+1)': probs['Raw_Count(1)'],
        'Direction': direction,
        'Entropy': entropy,
        'Gini': gini,
        'Classification_Error': class_error,
        'Confidence': confidence
    }).reset_index(drop=True)
    
    cluster_summary.sort_values('Direction', inplace=True)
    
    remap = {old_id: new_id for new_id, old_id in enumerate(cluster_summary['Cluster_ID'], 1)}
    cluster_summary['Cluster_ID'] = cluster_summary['Cluster_ID'].map(remap)
    states_df['Cluster'] = states_df['Cluster'].map(remap)
    
    cluster_summary.sort_values('Direction', inplace=True)
    
    cluster_members = states_df.copy()
    cluster_members.reset_index(inplace=True)
    cluster_members = cluster_members[['Cluster', 'State_Tuple', 'State_Count', 'P(-1)', 'P(0)', 'P(+1)', 'Direction', 'Entropy', 'Gini', 'Classification_Error']]
    cluster_members.rename(columns={'State_Tuple': 'State'}, inplace=True)
    cluster_members.sort_values(['Cluster', 'State'], inplace=True)
    
    return cluster_summary, cluster_members

def make_visualizations(summary_dict, dendrograms, trees):
    print("Generating visualizations...")
    os.makedirs('plots', exist_ok=True)
    
    for name, (summary, _) in summary_dict.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(summary[['P(-1)', 'P(0)', 'P(+1)']].set_index(summary['Cluster_ID']), 
                    annot=True, cmap='RdYlGn', center=0.33)
        plt.title(f'{name} Transition Probabilities')
        plt.ylabel('Cluster (Ordered by Direction)')
        plt.savefig(f'plots/heatmap_{name}.png')
        plt.close()
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Cluster_ID', y='N_Observations', data=summary, palette='viridis')
        plt.title(f'{name} Cluster Size')
        plt.savefig(f'plots/size_{name}.png')
        plt.close()
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x='Cluster_ID', y='Direction', data=summary, palette='RdYlGn')
        plt.title(f'{name} Direction by Cluster')
        plt.axhline(0, color='black', linestyle='--')
        plt.savefig(f'plots/direction_{name}.png')
        plt.close()
        
    from scipy.cluster.hierarchy import dendrogram
    for name, Z in dendrograms.items():
        plt.figure(figsize=(10, 7))
        dendrogram(Z)
        plt.title(f'{name} Dendrogram')
        plt.savefig(f'plots/dendrogram_{name}.png')
        plt.close()
        
    for name, clf in trees.items():
        plt.figure(figsize=(15, 10))
        feature_names = ['Direction', 'Entropy' if 'entropy' in name.lower() else 'Gini']
        plot_tree(clf, feature_names=feature_names, filled=True, class_names=['-1', '0', '+1'])
        plt.title(f'{name} Decision Tree')
        plt.savefig(f'plots/tree_{name}.png')
        plt.close()

def main():
    df = download_data()
    df = prepare_data(df)
    df, state_cols = construct_states(df, lookback=5)
    states_df = compute_transitions(df)
    
    states_df.to_csv('state_transition_table.csv')
    
    labels_js, k_js, score_js, Z_js = cluster_js(states_df)
    labels_sse, k_sse, score_sse, Z_sse = cluster_sse(states_df)
    labels_ent, depth_ent, score_ent, clf_ent = tree_clustering(states_df.copy(), df, 'entropy')
    labels_gini, depth_gini, score_gini, clf_gini = tree_clustering(states_df.copy(), df, 'gini')
    
    sum_js, mem_js = reestimate_clusters(states_df.copy(), labels_js)
    sum_sse, mem_sse = reestimate_clusters(states_df.copy(), labels_sse)
    sum_ent, mem_ent = reestimate_clusters(states_df.copy(), labels_ent)
    sum_gini, mem_gini = reestimate_clusters(states_df.copy(), labels_gini)
    
    sum_js.to_csv('js_cluster_summary.csv', index=False)
    sum_sse.to_csv('sse_cluster_summary.csv', index=False)
    sum_ent.to_csv('entropy_tree_summary.csv', index=False)
    sum_gini.to_csv('gini_tree_summary.csv', index=False)
    
    mem_js.to_csv('cluster_members_js.csv', index=False)
    mem_sse.to_csv('cluster_members_sse.csv', index=False)
    mem_ent.to_csv('cluster_members_entropy.csv', index=False)
    mem_gini.to_csv('cluster_members_gini.csv', index=False)
    
    comp_data = [
        ['Jensen-Shannon', len(sum_js), sum_js['N_Observations'].mean(), sum_js['Entropy'].mean(), sum_js['Confidence'].mean(), sum_js['Direction'].mean(), score_js, np.nan],
        ['SSE', len(sum_sse), sum_sse['N_Observations'].mean(), sum_sse['Entropy'].mean(), sum_sse['Confidence'].mean(), sum_sse['Direction'].mean(), score_sse, np.nan],
        ['Entropy Tree', len(sum_ent), sum_ent['N_Observations'].mean(), sum_ent['Entropy'].mean(), sum_ent['Confidence'].mean(), sum_ent['Direction'].mean(), np.nan, score_ent],
        ['Gini Tree', len(sum_gini), sum_gini['N_Observations'].mean(), sum_gini['Entropy'].mean(), sum_gini['Confidence'].mean(), sum_gini['Direction'].mean(), np.nan, score_gini]
    ]
    comp_df = pd.DataFrame(comp_data, columns=['Method', 'Number_of_Clusters', 'Average_Cluster_Size', 'Average_Entropy', 'Average_Confidence', 'Average_Direction', 'Silhouette_Score', 'Cross_Validation_Score'])
    comp_df.to_csv('comparison_report.csv', index=False)
    
    summary_dict = {
        'JS': (sum_js, mem_js),
        'SSE': (sum_sse, mem_sse),
        'Entropy_Tree': (sum_ent, mem_ent),
        'Gini_Tree': (sum_gini, mem_gini)
    }
    dendrograms = {'JS': Z_js, 'SSE': Z_sse}
    trees = {'Entropy': clf_ent, 'Gini': clf_gini}
    
    make_visualizations(summary_dict, dendrograms, trees)
    print("Done! All outputs generated.")

if __name__ == '__main__':
    main()
