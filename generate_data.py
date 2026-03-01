import pandas as pd
import numpy as np
import networkx as nx
import os

def generate_complex_data(num_students=2000):
    np.random.seed(42)
    
    # 1. Generate Peer Network (Graph Theory)
    # Using a Barabasi-Albert graph to simulate real-world social networks (study groups)
    G = nx.barabasi_albert_graph(num_students, 3)
    
    # 2. Base metrics
    data = []
    for i in range(num_students):
        # NLP & Linguistic Metrics
        semantic_fatigue = np.random.beta(2, 5) # 0 to 1 score (Linguistic simplification)
        help_seeking_drop = np.random.uniform(0, 100) # % drop in asking questions
        forum_sentiment = np.random.normal(0, 0.5)
        
        # LMS Metrics
        lms_logins = int(np.random.poisson(10))
        assignment_delay = round(np.random.exponential(2), 1)
        
        # Determine Base Burnout (Hidden Variable)
        base_burnout = (semantic_fatigue * 30) + (help_seeking_drop * 0.3) + (assignment_delay * 2) - (forum_sentiment * 10)
        
        data.append({
            'student_id': i,
            'semantic_fatigue_score': round(semantic_fatigue, 3),
            'help_seeking_drop_pct': round(help_seeking_drop, 1),
            'forum_sentiment': round(max(-1, min(1, forum_sentiment)), 2),
            'lms_logins': lms_logins,
            'assignment_delay_days': assignment_delay,
            'base_burnout': base_burnout
        })
        
    df = pd.DataFrame(data)
    
    # 3. Graph Contagion Effect (The "Wow" Factor)
    # If a student's friends have high burnout, their burnout increases
    peer_contagion_scores = []
    final_risk_scores = []
    
    for i in range(num_students):
        friends = list(G.neighbors(i))
        if friends:
            friend_burnout_avg = df.loc[friends, 'base_burnout'].mean()
        else:
            friend_burnout_avg = df['base_burnout'].mean()
            
        # Add contagion metric
        contagion_risk = friend_burnout_avg * 0.4
        peer_contagion_scores.append(round(contagion_risk, 2))
        
        # Calculate final 0-100 score
        final_score = df.loc[i, 'base_burnout'] + contagion_risk
        final_risk_scores.append(min(100, max(0, final_score + np.random.normal(0, 5))))
        
    df['peer_contagion_risk'] = peer_contagion_scores
    df['final_burnout_score'] = [round(x, 1) for x in final_risk_scores]
    
    # Classify
    df['risk_stage'] = pd.cut(df['final_burnout_score'], bins=[-1, 35, 70, 101], labels=['Stage 1: Stable', 'Stage 2: Drifting', 'Stage 3: High Risk'])
    
    # Drop the hidden calculation variable
    df = df.drop(columns=['base_burnout'])
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/complex_student_data.csv', index=False)
    
    # Save network edges for the dashboard
    nx.write_edgelist(G, "data/peer_network.edgelist", data=False)
    print("Complex dataset and Graph network generated successfully.")

if __name__ == "__main__":
    generate_complex_data()