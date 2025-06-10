import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import base64
from itertools import combinations

# Set page config
st.set_page_config(page_title="Project Scheduling & Crashing", layout="wide")

# Initialize session state
if 'activities' not in st.session_state:
    st.session_state.activities = []
if 'network_structure' not in st.session_state:
    st.session_state.network_structure = []
if 'indirect_cost_rate' not in st.session_state:
    st.session_state.indirect_cost_rate = 0

def create_network_diagram(activities, title="Network Diagram", critical_path=None):
    """Create network diagram with activities"""
    G = nx.DiGraph()
    
    # Add nodes for activities
    for activity in activities:
        G.add_node(activity['Activity'], 
                  duration=activity.get('Duration (days)', activity.get('Normal Duration', 0)),
                  cost=activity.get('Normal Cost', 0))
    
    # Add edges based on predecessors
    for activity in activities:
        if activity.get('Predecessors') and activity['Predecessors'] != '-':
            predecessors = str(activity['Predecessors']).split(',')
            for pred in predecessors:
                pred = pred.strip()
                if pred and pred != '-':
                    G.add_edge(pred, activity['Activity'])
    
    # Create layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if critical_path and node in critical_path:
            node_colors.append('red')
        else:
            node_colors.append('lightblue')
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=1000, ax=ax)
    
    # Draw edges
    edge_colors = []
    for edge in G.edges():
        if critical_path and edge[0] in critical_path and edge[1] in critical_path:
            edge_colors.append('red')
        else:
            edge_colors.append('black')
    
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, 
                          arrows=True, arrowsize=20, ax=ax)
    
    # Draw labels
    labels = {}
    for node in G.nodes():
        activity_data = next((a for a in activities if a['Activity'] == node), None)
        if activity_data:
            duration = activity_data.get('Duration (days)', activity_data.get('Normal Duration', 0))
            labels[node] = f"{node}\n({duration}d)"
        else:
            labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    return fig

def calculate_critical_path(activities):
    """Calculate critical path using CPM"""
    # Create adjacency list
    adj_list = {}
    durations = {}
    
    for activity in activities:
        act_name = activity['Activity']
        duration = activity.get('Duration (days)', activity.get('Normal Duration', 0))
        durations[act_name] = duration
        adj_list[act_name] = []
        
        if activity.get('Predecessors') and activity['Predecessors'] != '-':
            predecessors = str(activity['Predecessors']).split(',')
            for pred in predecessors:
                pred = pred.strip()
                if pred and pred != '-':
                    if pred not in adj_list:
                        adj_list[pred] = []
                    adj_list[pred].append(act_name)
    
    # Forward pass
    early_start = {}
    early_finish = {}
    
    def forward_pass(node):
        if node in early_start:
            return early_start[node], early_finish[node]
        
        max_ef = 0
        # Find predecessors
        predecessors = []
        for act in activities:
            if act['Activity'] == node and act.get('Predecessors') and act['Predecessors'] != '-':
                predecessors = str(act['Predecessors']).split(',')
                break
        
        for pred in predecessors:
            pred = pred.strip()
            if pred and pred != '-':
                _, pred_ef = forward_pass(pred)
                max_ef = max(max_ef, pred_ef)
        
        early_start[node] = max_ef
        early_finish[node] = max_ef + durations.get(node, 0)
        return early_start[node], early_finish[node]
    
    # Calculate for all activities
    for activity in activities:
        forward_pass(activity['Activity'])
    
    # Find project duration
    project_duration = max(early_finish.values()) if early_finish else 0
    
    # Backward pass
    late_start = {}
    late_finish = {}
    
    # Initialize end activities
    for activity in activities:
        act_name = activity['Activity']
        if act_name not in adj_list or not adj_list[act_name]:
            late_finish[act_name] = early_finish[act_name]
            late_start[act_name] = late_finish[act_name] - durations[act_name]
    
    def backward_pass(node):
        if node in late_start:
            return late_start[node], late_finish[node]
        
        min_ls = float('inf')
        successors = adj_list.get(node, [])
        
        for succ in successors:
            succ_ls, _ = backward_pass(succ)
            min_ls = min(min_ls, succ_ls)
        
        if min_ls == float('inf'):
            min_ls = project_duration
        
        late_finish[node] = min_ls
        late_start[node] = min_ls - durations.get(node, 0)
        return late_start[node], late_finish[node]
    
    # Calculate for all activities
    for activity in activities:
        backward_pass(activity['Activity'])
    
    # Find critical path
    critical_activities = []
    for activity in activities:
        act_name = activity['Activity']
        if (early_start.get(act_name, 0) == late_start.get(act_name, 0) and 
            early_finish.get(act_name, 0) == late_finish.get(act_name, 0)):
            critical_activities.append(act_name)
    
    return critical_activities, project_duration, early_start, early_finish, late_start, late_finish

def calculate_direct_cost(activities, crash_states):
    """Calculate direct cost based on current crash states"""
    total_direct_cost = 0
    
    for activity in activities:
        act_name = activity['Activity']
        normal_cost = activity.get('Normal Cost', 0)
        normal_duration = activity.get('Normal Duration', 0)
        crash_duration = activity.get('Crash Duration', normal_duration)
        crash_cost = activity.get('Crash Cost', normal_cost)
        
        # Get current crashed days for this activity
        crashed_days = crash_states.get(act_name, 0)
        
        if normal_duration > crash_duration and crashed_days > 0:
            # Calculate slope
            slope = (crash_cost - normal_cost) / (normal_duration - crash_duration)
            activity_cost = normal_cost + (crashed_days * slope)
        else:
            activity_cost = normal_cost
        
        total_direct_cost += activity_cost
    
    return total_direct_cost

def get_all_crash_combinations(activities):
    """Generate all possible crash combinations"""
    crashable_activities = []
    
    for activity in activities:
        act_name = activity['Activity']
        normal_duration = activity.get('Normal Duration', 0)
        crash_duration = activity.get('Crash Duration', normal_duration)
        
        if normal_duration > crash_duration:
            max_crash = normal_duration - crash_duration
            crashable_activities.append((act_name, max_crash))
    
    # Generate all possible combinations
    all_combinations = [{}]  # Start with no crashing
    
    for act_name, max_crash in crashable_activities:
        new_combinations = []
        for existing_combo in all_combinations:
            # Add combinations with this activity crashed 1 to max_crash days
            for crash_days in range(1, max_crash + 1):
                new_combo = existing_combo.copy()
                new_combo[act_name] = crash_days
                new_combinations.append(new_combo)
        all_combinations.extend(new_combinations)
    
    return all_combinations

def perform_comprehensive_crashing(activities, indirect_cost_rate, target_duration=None):
    """Perform comprehensive project crashing with all combinations"""
    
    # Get all possible crash combinations
    all_combinations = get_all_crash_combinations(activities)
    
    results = []
    
    for combo in all_combinations:
        # Create modified activities with crashed durations
        modified_activities = []
        for activity in activities:
            new_activity = activity.copy()
            act_name = activity['Activity']
            normal_duration = activity.get('Normal Duration', 0)
            crashed_days = combo.get(act_name, 0)
            
            # Set current duration
            new_activity['Duration (days)'] = normal_duration - crashed_days
            modified_activities.append(new_activity)
        
        # Calculate critical path for this combination
        critical_path, project_duration, es, ef, ls, lf = calculate_critical_path(modified_activities)
        
        # Calculate costs
        direct_cost = calculate_direct_cost(activities, combo)
        indirect_cost = project_duration * indirect_cost_rate
        total_cost = direct_cost + indirect_cost
        
        # Create crashed activities description
        crashed_activities = [(act, days) for act, days in combo.items() if days > 0]
        
        result = {
            'Crash Combination': combo.copy(),
            'Project Duration': project_duration,
            'Direct Cost': direct_cost,
            'Indirect Cost': indirect_cost,
            'Total Cost': total_cost,
            'Critical Path': critical_path.copy(),
            'Activities': modified_activities.copy(),
            'Crashed Activities': crashed_activities
        }
        
        results.append(result)
        
        # If target duration is specified and reached, we can continue to find all solutions
        # but we'll mark which ones meet the target
        if target_duration and project_duration <= target_duration:
            result['Meets Target'] = True
        else:
            result['Meets Target'] = False
    
    # Sort results by total cost, then by project duration
    results.sort(key=lambda x: (x['Total Cost'], x['Project Duration']))
    
    return results

# Main App
st.title("🏗️ Project Scheduling & Crashing Analysis")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", 
                           ["Data Input", "Network Analysis", "Crashing Analysis"])

if page == "Data Input":
    st.header("📊 Activity Data Input")
    
    # Template download
    st.subheader("Template Excel")
    template_data = {
        'Activity': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'Description': ['Foundation', 'Frame', 'Walls', 'Roof', 'Finishing', 'Plumbing', 'Electrical'],
        'Predecessors': ['-', 'A', 'A', 'B,C', 'D', 'C', 'E,F'],
        'Normal Duration': [3, 6, 10, 11, 8, 5, 6],
        'Crash Duration': [2, 4, 9, 7, 6, 5, 6],
        'Normal Cost': [50, 80, 60, 50, 100, 40, 70],
        'Crash Cost': [70, 160, 90, 150, 160, 70, 70]
    }
    template_df = pd.DataFrame(template_data)
    
    st.write("Download template Excel file:")
    
    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        template_df.to_excel(writer, index=False, sheet_name='Activities')
    
    excel_data = output.getvalue()
    
    st.download_button(
        label="📥 Download Template Excel",
        data=excel_data,
        file_name="activity_template.xlsx",
        mime="application/vnd.openxl.sheet"
    )
    
    st.write("Template format:")
    st.dataframe(template_df)
    
    # Calculate and display crash slopes for template
    st.write("**Crash Slopes for Template Data:**")
    slope_data = []
    for _, row in template_df.iterrows():
        if row['Normal Duration'] > row['Crash Duration']:
            slope = (row['Crash Cost'] - row['Normal Cost']) / (row['Normal Duration'] - row['Crash Duration'])
            max_crash = row['Normal Duration'] - row['Crash Duration']
        else:
            slope = 0
            max_crash = 0
        
        slope_data.append({
            'Activity': row['Activity'],
            'Slope': slope,
            'Max Crash Time': max_crash
        })
    
    slope_df = pd.DataFrame(slope_data)
    st.dataframe(slope_df)
    
    # File upload
    st.subheader("Upload Excel/CSV File")
    uploaded_file = st.file_uploader("Choose Excel or CSV file", type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file is not None:
        try:
            # Read file based on extension
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Validate columns
            required_columns = ['Activity', 'Predecessors', 'Normal Duration', 'Crash Duration', 'Normal Cost', 'Crash Cost']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                st.success("File uploaded successfully!")
                st.dataframe(df)
                
                if st.button("Load Activities from File"):
                    # Clear existing activities
                    st.session_state.activities = []
                    
                    # Add activities from file
                    for _, row in df.iterrows():
                        activity = {
                            'Activity': str(row['Activity']),
                            'Description': str(row.get('Description', '')),
                            'Predecessors': str(row['Predecessors']) if pd.notna(row['Predecessors']) else '-',
                            'Normal Duration': int(row['Normal Duration']) if pd.notna(row['Normal Duration']) else 0,
                            'Crash Duration': int(row['Crash Duration']) if pd.notna(row['Crash Duration']) else 0,
                            'Normal Cost': float(row['Normal Cost']) if pd.notna(row['Normal Cost']) else 0,
                            'Crash Cost': float(row['Crash Cost']) if pd.notna(row['Crash Cost']) else 0
                        }
                        st.session_state.activities.append(activity)
                    
                    st.success(f"Loaded {len(st.session_state.activities)} activities from file!")
                    st.rerun()
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    st.markdown("---")
    
    # Manual input
    st.subheader("Manual Activity Input")
    
    with st.form("activity_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            activity_name = st.text_input("Activity Name")
            description = st.text_input("Description")
            predecessors = st.text_input("Predecessors (comma separated, '-' for none)")
            normal_duration = st.number_input("Normal Duration (days)", min_value=1, value=1)
        
        with col2:
            crash_duration = st.number_input("Crash Duration (days)", min_value=1, value=1)
            normal_cost = st.number_input("Normal Cost", min_value=0, value=0)
            crash_cost = st.number_input("Crash Cost", min_value=0, value=0)
        
        submitted = st.form_submit_button("Add Activity")
        
        if submitted and activity_name:
            new_activity = {
                'Activity': activity_name,
                'Description': description,
                'Predecessors': predecessors,
                'Normal Duration': normal_duration,
                'Crash Duration': crash_duration,
                'Normal Cost': normal_cost,
                'Crash Cost': crash_cost
            }
            st.session_state.activities.append(new_activity)
            st.success(f"Activity {activity_name} added!")
    
    # Display current activities
    if st.session_state.activities:
        st.subheader("Current Activities")
        activities_df = pd.DataFrame(st.session_state.activities)
        st.dataframe(activities_df)
        
        # Clear activities
        if st.button("Clear All Activities"):
            st.session_state.activities = []
            st.rerun()
    
    # Indirect cost input
    st.subheader("Indirect Cost")
    st.session_state.indirect_cost_rate = st.number_input(
        "Indirect Cost Rate (per day)", 
        min_value=0.0, 
        value=float(st.session_state.indirect_cost_rate)
    )

elif page == "Network Analysis":
    st.header("🔗 Network Structure Analysis")
    
    if not st.session_state.activities:
        st.warning("Please add activities in the Data Input page first.")
    else:
        # Calculate critical path
        critical_path, project_duration, es, ef, ls, lf = calculate_critical_path(st.session_state.activities)
        
        # Display project info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Project Duration", f"{project_duration} days")
        with col2:
            st.metric("Critical Activities", len(critical_path))
        with col3:
            st.metric("Total Activities", len(st.session_state.activities))
        
        # Network diagram
        st.subheader("Network Diagram")
        fig = create_network_diagram(st.session_state.activities, 
                                   "Initial Network Diagram", 
                                   critical_path)
        st.pyplot(fig)
        
        # Critical path analysis
        st.subheader("Critical Path Analysis")
        cpm_data = []
        for activity in st.session_state.activities:
            act_name = activity['Activity']
            duration = activity.get('Normal Duration', 0)
            total_float = ls.get(act_name, 0) - es.get(act_name, 0)
            
            cpm_data.append({
                'Activity': act_name,
                'Duration': duration,
                'Early Start': es.get(act_name, 0),
                'Early Finish': ef.get(act_name, 0),
                'Late Start': ls.get(act_name, 0),
                'Late Finish': lf.get(act_name, 0),
                'Total Float': total_float,
                'Critical': 'Yes' if act_name in critical_path else 'No'
            })
        
        cpm_df = pd.DataFrame(cpm_data)
        st.dataframe(cpm_df)
        
        # Critical path display
        st.subheader("Critical Path")
        st.write("Critical Path: " + " → ".join(critical_path))

elif page == "Crashing Analysis":
    st.header("⚡ Project Crashing Analysis")
    
    if not st.session_state.activities:
        st.warning("Please add activities in the Data Input page first.")
    elif st.session_state.indirect_cost_rate <= 0:
        st.warning("Please set indirect cost rate in the Data Input page.")
    else:
        # Crashing options
        st.subheader("Crashing Options")
        crash_option = st.radio("Select crashing approach:", 
                               ["Comprehensive Analysis", "Target Duration"])
        
        target_duration = None
        if crash_option == "Target Duration":
            critical_path, original_duration, _, _, _, _ = calculate_critical_path(st.session_state.activities)
            target_duration = st.number_input("Target Duration (days)", 
                                            min_value=1, 
                                            max_value=original_duration-1, 
                                            value=max(1, original_duration-5))
        
        if st.button("Perform Crashing Analysis"):
            # Perform comprehensive crashing
            results = perform_comprehensive_crashing(st.session_state.activities, 
                                                   st.session_state.indirect_cost_rate,
                                                   target_duration)
            
            # Display results
            st.subheader("Crashing Results")
            
            # Summary table - show top 20 results
            st.write("**Top 20 Solutions (sorted by Total Cost):**")
            summary_data = []
            for i, result in enumerate(results[:20]):
                crashed_activities = ", ".join([f"{act}(-{days})" for act, days in result['Crashed Activities']]) if result['Crashed Activities'] else "None"
                summary_data.append({
                    'Rank': i + 1,
                    'Duration': result['Project Duration'],
                    'Direct Cost': f"${result['Direct Cost']:,.2f}",
                    'Indirect Cost': f"${result['Indirect Cost']:,.2f}",
                    'Total Cost': f"${result['Total Cost']:,.2f}",
                    'Critical Path': ' → '.join(result['Critical Path']),
                    'Crashed Activities': crashed_activities
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)
            
            # Find optimum
            optimum_result = results[0]  # First result is the minimum cost
            
            st.success("Optimum solution found!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Optimum Duration", f"{optimum_result['Project Duration']} days")
            with col2:
                st.metric("Optimum Total Cost", f"${optimum_result['Total Cost']:,.2f}")
            with col3:
                st.metric("Direct Cost", f"${optimum_result['Direct Cost']:,.2f}")
            
            # Show optimum solution details
            st.subheader("Optimum Solution Details")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Cost Breakdown:**")
                st.write(f"Direct Cost: ${optimum_result['Direct Cost']:,.2f}")
                st.write(f"Indirect Cost: ${optimum_result['Indirect Cost']:,.2f}")
                st.write(f"Total Cost: ${optimum_result['Total Cost']:,.2f}")
            
            with col2:
                st.write("**Critical Path:**")
                st.write(" → ".join(optimum_result['Critical Path']))
                if optimum_result['Crashed Activities']:
                    st.write("**Crashed Activities:**")
                    for act, days in optimum_result['Crashed Activities']:
                        st.write(f"- {act}: {days} day(s)")
                else:
                    st.write("**No activities crashed**")
            
            # Network diagrams comparison
            st.subheader("Network Diagrams Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Initial Network (Before Crashing)**")
                initial_critical_path, _, _, _, _, _ = calculate_critical_path(st.session_state.activities)
                initial_fig = create_network_diagram(st.session_state.activities,
                                                   "Initial Network",
                                                   initial_critical_path)
                st.pyplot(initial_fig)
            
            with col2:
                st.write("**Optimum Network (After Crashing)**")
                opt_fig = create_network_diagram(optimum_result['Activities'],
                                               "Optimum Network",
                                               optimum_result['Critical Path'])
                st.pyplot(opt_fig)
            
            # Cost analysis chart
            st.subheader("Cost Analysis Chart")
            
            # Group results by duration for plotting
            duration_groups = {}
            for result in results:
                duration = result['Project Duration']
                if duration not in duration_groups:
                    duration_groups[duration] = []
                duration_groups[duration].append(result)
            
            # Get minimum cost for each duration
            plot_data = []
            for duration in sorted(duration_groups.keys()):
                min_cost_result = min(duration_groups[duration], key=lambda x: x['Total Cost'])
                plot_data.append({
                    'Duration': duration,
                    'Direct Cost': min_cost_result['Direct Cost'],
                    'Indirect Cost': min_cost_result['Indirect Cost'],
                    'Total Cost': min_cost_result['Total Cost']
                })
            
            if len(plot_data) > 1:
                cost_fig, ax = plt.subplots(figsize=(10, 6))
                
                durations = [d['Duration'] for d in plot_data]
                direct_costs = [d['Direct Cost'] for d in plot_data]
                indirect_costs = [d['Indirect Cost'] for d in plot_data]
                total_costs = [d['Total Cost'] for d in plot_data]
                
                ax.plot(durations, direct_costs, 'b-o', label='Direct Cost')
                ax.plot(durations, indirect_costs, 'r-o', label='Indirect Cost')
                ax.plot(durations, total_costs, 'g-o', label='Total Cost', linewidth=2)
                
                # Mark optimum point
                opt_duration = optimum_result['Project Duration']
                opt_cost = optimum_result['Total Cost']
                ax.plot(opt_duration, opt_cost, 'go', markersize=10, label='Optimum')
                
                ax.set_xlabel('Project Duration (days)')
                ax.set_ylabel('Cost ($)')
                ax.set_title('Project Crashing Cost Analysis')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(cost_fig)
            
            # Show details for target duration if specified
            if target_duration:
                st.subheader(f"Solutions Meeting Target Duration ({target_duration} days)")
                target_solutions = [r for r in results if r['Meets Target']]
                
                if target_solutions:
                    st.write(f"Found {len(target_solutions)} solution(s) meeting target duration:")
                    
                    target_summary = []
                    for i, result in enumerate(target_solutions[:10]):  # Show top 10
                        crashed_activities = ", ".join([f"{act}(-{days})" for act, days in result['Crashed Activities']]) if result['Crashed Activities'] else "None"
                        target_summary.append({
                            'Rank': i + 1,
                            'Duration': result['Project Duration'],
                            'Total Cost': f"${result['Total Cost']:,.2f}",
                            'Crashed Activities': crashed_activities
                        })
                    
                    target_df = pd.DataFrame(target_summary)
                    st.dataframe(target_df)
                else:
                    st.warning(f"No solutions found that meet target duration of {target_duration} days.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Project Scheduling & Crashing Tool**")
st.sidebar.markdown("Built with Streamlit")