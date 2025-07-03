import os
import json
import argparse
import networkx as nx
import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, Counter
import re
import plotly.graph_objects as go
from dataclasses import dataclass, asdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from storage import OptimizedGraphStorage


@dataclass
class GraphNode:
    """Represents a node in the dependency graph"""
    id: str
    name: str
    node_type: str  # 'file', 'class', 'function', 'module'
    file_path: str
    language: str
    symbol_count: int = 0
    token_count: int = 0
    complexity_score: float = 0.0
    centrality_score: float = 0.0
    importance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    """Represents an edge in the dependency graph"""
    source: str
    target: str
    edge_type: str  # 'imports', 'calls', 'inherits', 'defines'
    weight: float = 1.0
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RepoGraphBuilder:
    """Builds high-level dependency graphs from parsed repository data"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.language_stats = Counter()
        self.file_dependencies = defaultdict(set)
        self.symbol_dependencies = defaultdict(set)
        self.module_map = {}  # Map files to modules
        
        # Enhanced import patterns for different languages
        self.import_patterns = {
            'python': [
                r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',
                r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
            ],
            'javascript': [
                r'import.*from\s+[\'"]([^\'\"]+)[\'"]',
                r'require\([\'"]([^\'\"]+)[\'"]\)',
                r'import\s+[\'"]([^\'\"]+)[\'"]',
            ],
            'typescript': [
                r'import.*from\s+[\'"]([^\'\"]+)[\'"]',
                r'import\s+[\'"]([^\'\"]+)[\'"]',
            ],
            'go': [
                r'import\s+[\'"]([^\'\"]+)[\'"]',
                r'import\s+\(\s*[\'"]([^\'\"]+)[\'"]',
            ],
            'java': [
                r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*);',
                r'import\s+static\s+([a-zA-Z_][a-zA-Z0-9_.]*);',
            ],
            'rust': [
                r'use\s+([a-zA-Z_][a-zA-Z0-9_:]*);',
                r'extern\s+crate\s+([a-zA-Z_][a-zA-Z0-9_]*);',
            ],
            'cpp': [
                r'#include\s+[<"]([^>"]+)[>"]',
            ],
            'c': [
                r'#include\s+[<"]([^>"]+)[>"]',
            ],
        }
        
        # Function call patterns
        self.call_patterns = {
            'python': r'([a-zA-Z_][a-zA-Z0-9_.]*)\s*\(',
            'javascript': r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(',
            'typescript': r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(',
            'java': r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            'go': r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            'rust': r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
        }

    def load_parsed_data(self, data_path: str) -> pd.DataFrame:
        """Load parsed repository data from various formats"""
        print(f"📁 Loading parsed data from: {data_path}")
        
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        print(f"✅ Loaded {len(df)} chunks from {df['file_path'].nunique()} files")
        return df

    def build_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """Build comprehensive dependency graph from parsed data"""
        print("🏗️  Building repository dependency graph...")
        
        # Step 1: Build file-level nodes
        self._build_file_nodes(df)
        
        # Step 2: Build symbol-level nodes (classes, functions)
        self._build_symbol_nodes(df)
        
        # Step 3: Extract dependencies from imports and content
        self._extract_dependencies(df)
        
        # Step 4: Calculate graph metrics
        self._calculate_metrics()
        
        # Step 5: Add nodes and edges to NetworkX graph
        self._build_networkx_graph()
        
        print(f"✅ Graph built: {len(self.nodes)} nodes, {len(self.edges)} edges")
        return self.graph

    def _build_file_nodes(self, df: pd.DataFrame):
        """Create file-level nodes"""
        print("📄 Building file nodes...")
        
        file_stats = df.groupby('file_path').agg({
            'language': 'first',
            'token_count': 'sum',
            'symbol_type': 'count',
            'last_modified': 'first'
        }).reset_index()
        
        for _, row in file_stats.iterrows():
            file_path = row['file_path']
            node_id = f"file:{file_path}"
            
            # Calculate complexity based on token count and symbol diversity
            symbol_diversity = df[df['file_path'] == file_path]['symbol_type'].nunique()
            complexity = (row['token_count'] / 1000) + (symbol_diversity / 10)
            
            node = GraphNode(
                id=node_id,
                name=os.path.basename(file_path),
                node_type='file',
                file_path=file_path,
                language=row['language'],
                symbol_count=row['symbol_type'],
                token_count=row['token_count'],
                complexity_score=complexity
            )
            
            self.nodes[node_id] = node
            self.language_stats[row['language']] += 1

    def _build_symbol_nodes(self, df: pd.DataFrame):
        """Create symbol-level nodes (classes, functions, etc.)"""
        print("🔧 Building symbol nodes...")
        
        # Filter meaningful symbols (skip small assignments, etc.)
        meaningful_symbols = df[
            (df['symbol_type'].isin([
                'function_definition', 'class_definition', 'method_definition',
                'function_declaration', 'class_declaration', 'interface_declaration'
            ])) & 
            (df['token_count'] >= 20)  # Filter out tiny symbols
        ].copy()
        
        for _, row in meaningful_symbols.iterrows():
            if pd.isna(row['symbol_name']) or not row['symbol_name']:
                continue
                
            symbol_name = str(row['symbol_name']).strip()
            if not symbol_name:
                continue
                
            node_id = f"symbol:{row['file_path']}:{symbol_name}"
            
            # Calculate symbol complexity
            complexity = (row['token_count'] / 100) + (1 if row['parent_class'] else 0)
            
            node = GraphNode(
                id=node_id,
                name=symbol_name,
                node_type=row['symbol_type'],
                file_path=row['file_path'],
                language=row['language'],
                token_count=row['token_count'],
                complexity_score=complexity
            )
            
            self.nodes[node_id] = node
            
            # Create edge from file to symbol
            file_node_id = f"file:{row['file_path']}"
            if file_node_id in self.nodes:
                edge = GraphEdge(
                    source=file_node_id,
                    target=node_id,
                    edge_type='defines',
                    weight=1.0
                )
                self.edges.append(edge)

    def _extract_dependencies(self, df: pd.DataFrame):
        """Extract import and call dependencies"""
        print("🔗 Extracting dependencies...")
        
        # Group by file for efficient processing
        file_groups = df.groupby('file_path')
        
        for file_path, group in tqdm(file_groups, desc="Processing files"):
            language = group['language'].iloc[0]
            self._process_file_dependencies(file_path, group, language)

    def _process_file_dependencies(self, file_path: str, group: pd.DataFrame, language: str):
        """Process dependencies for a single file"""
        file_node_id = f"file:{file_path}"
        all_content = '\n'.join(group['content'].dropna().astype(str))
        
        # Extract imports
        imports = self._extract_imports(all_content, language)
        for imported_module in imports:
            target_file = self._resolve_import_to_file(imported_module, file_path, language)
            if target_file:
                target_node_id = f"file:{target_file}"
                if target_node_id in self.nodes:
                    edge = GraphEdge(
                        source=file_node_id,
                        target=target_node_id,
                        edge_type='imports',
                        weight=1.0,
                        confidence=0.8
                    )
                    self.edges.append(edge)
        
        # Extract function calls (simplified)
        calls = self._extract_function_calls(all_content, language)
        for call in calls[:10]:  # Limit to avoid noise
            # Try to find the called function in other files
            for target_file, target_group in group.groupby('file_path'):
                if target_file == file_path:
                    continue
                    
                matching_symbols = target_group[
                    target_group['symbol_name'].astype(str).str.contains(call, na=False, regex=False)
                ]
                
                if not matching_symbols.empty:
                    target_node_id = f"file:{target_file}"
                    if target_node_id in self.nodes:
                        edge = GraphEdge(
                            source=file_node_id,
                            target=target_node_id,
                            edge_type='calls',
                            weight=0.5,
                            confidence=0.6
                        )
                        self.edges.append(edge)
                        break

    def _extract_imports(self, content: str, language: str) -> List[str]:
        """Extract import statements from code content"""
        imports = []
        patterns = self.import_patterns.get(language, [])
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            imports.extend(matches)
        
        # Clean and filter imports
        cleaned_imports = []
        for imp in imports:
            if imp and len(imp) > 1 and not imp.startswith('.'):
                cleaned_imports.append(imp.strip())
        
        return list(set(cleaned_imports))  # Remove duplicates

    def _extract_function_calls(self, content: str, language: str) -> List[str]:
        """Extract function calls from code content"""
        pattern = self.call_patterns.get(language)
        if not pattern:
            return []
        
        calls = re.findall(pattern, content)
        
        # Filter out common keywords and short names
        filtered_calls = []
        keywords = {'if', 'for', 'while', 'print', 'len', 'str', 'int', 'float', 'list', 'dict'}
        
        for call in calls:
            if (len(call) > 2 and 
                call.lower() not in keywords and 
                not call.startswith('_') and
                call.isalnum() or '_' in call):
                filtered_calls.append(call)
        
        return list(set(filtered_calls))

    def _resolve_import_to_file(self, import_name: str, current_file: str, language: str) -> Optional[str]:
        """Resolve import statement to actual file path"""
        current_dir = os.path.dirname(current_file)
        
        # Language-specific resolution
        if language == 'python':
            # Convert module.submodule to module/submodule.py
            possible_paths = [
                import_name.replace('.', '/') + '.py',
                import_name.replace('.', '/') + '/__init__.py',
                os.path.join(current_dir, import_name.replace('.', '/') + '.py'),
            ]
        elif language in ['javascript', 'typescript']:
            # Handle relative imports
            if import_name.startswith('./') or import_name.startswith('../'):
                base_path = os.path.join(current_dir, import_name)
                possible_paths = [
                    base_path + '.js',
                    base_path + '.ts',
                    base_path + '/index.js',
                    base_path + '/index.ts',
                ]
            else:
                return None  # External module
        else:
            return None
        
        # Check if any of the possible paths exist in our nodes
        for path in possible_paths:
            normalized_path = os.path.normpath(path)
            file_node_id = f"file:{normalized_path}"
            if file_node_id in self.nodes:
                return normalized_path
        
        return None

    def _calculate_metrics(self):
        """Calculate various graph metrics for nodes"""
        print("📊 Calculating graph metrics...")
        
        # Build temporary graph for centrality calculations
        temp_graph = nx.DiGraph()
        
        # Add nodes
        for node_id in self.nodes:
            temp_graph.add_node(node_id)
        
        # Add edges
        for edge in self.edges:
            temp_graph.add_edge(edge.source, edge.target, weight=edge.weight)
        
        # Calculate centrality measures
        try:
            in_centrality = nx.in_degree_centrality(temp_graph)
            out_centrality = nx.out_degree_centrality(temp_graph)
            
            for node_id, node in self.nodes.items():
                # Combine different centrality measures
                in_cent = in_centrality.get(node_id, 0)
                out_cent = out_centrality.get(node_id, 0)
                node.centrality_score = (in_cent + out_cent) / 2
                
                # Calculate importance score
                node.importance_score = (
                    node.complexity_score * 0.4 +
                    node.centrality_score * 0.4 +
                    (node.token_count / 1000) * 0.2
                )
                
        except Exception as e:
            print(f"⚠️ Warning: Could not calculate centrality metrics: {e}")

    def _build_networkx_graph(self):
        """Build the final NetworkX graph"""
        print("🔗 Building NetworkX graph...")
        
        # Add nodes with attributes
        for node_id, node in self.nodes.items():
            self.graph.add_node(node_id, **node.to_dict())
        
        # Add edges with attributes
        for edge in self.edges:
            self.graph.add_edge(
                edge.source, 
                edge.target, 
                **edge.to_dict()
            )

    def analyze_graph(self) -> Dict[str, Any]:
        """Perform comprehensive graph analysis"""
        print("🔍 Analyzing graph structure...")
        
        analysis = {
            'basic_stats': {
                'nodes': len(self.nodes),
                'edges': len(self.edges),
                'density': nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0,
                'is_connected': nx.is_weakly_connected(self.graph),
                'languages': dict(self.language_stats)
            },
            'node_types': {},
            'centrality': {},
            'complexity': {},
            'clusters': {}
        }
        
        # Node type distribution
        node_types = Counter(node.node_type for node in self.nodes.values())
        analysis['node_types'] = dict(node_types)
        
        # Top nodes by different metrics
        nodes_by_centrality = sorted(self.nodes.values(), key=lambda x: x.centrality_score, reverse=True)
        nodes_by_complexity = sorted(self.nodes.values(), key=lambda x: x.complexity_score, reverse=True)
        nodes_by_importance = sorted(self.nodes.values(), key=lambda x: x.importance_score, reverse=True)
        
        analysis['centrality']['top_central_nodes'] = [
            {'name': node.name, 'file': node.file_path, 'score': node.centrality_score}
            for node in nodes_by_centrality[:10]
        ]
        
        analysis['complexity']['top_complex_nodes'] = [
            {'name': node.name, 'file': node.file_path, 'score': node.complexity_score}
            for node in nodes_by_complexity[:10]
        ]
        
        analysis['complexity']['most_important_nodes'] = [
            {'name': node.name, 'file': node.file_path, 'score': node.importance_score}
            for node in nodes_by_importance[:10]
        ]
        
        # Find strongly connected components (circular dependencies)
        if len(self.graph) > 0:
            try:
                scc = list(nx.strongly_connected_components(self.graph))
                analysis['clusters']['circular_dependencies'] = [
                    list(component) for component in scc if len(component) > 1
                ]
            except:
                analysis['clusters']['circular_dependencies'] = []
        
        return analysis

    def export_graph(self, output_path: str, format: str = 'gexf'):
        """Export graph to various formats"""
        print(f"💾 Exporting graph to {format.upper()}: {output_path}")
        
        if format == 'gexf':
            nx.write_gexf(self.graph, output_path)
        elif format == 'json':
            data = {
                'nodes': [node.to_dict() for node in self.nodes.values()],
                'edges': [edge.to_dict() for edge in self.edges]
            }
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'graphml':
            nx.write_graphml(self.graph, output_path)
        elif format == 'csv':
            # Export as two CSV files: nodes and edges
            base_name = os.path.splitext(output_path)[0]
            
            nodes_df = pd.DataFrame([node.to_dict() for node in self.nodes.values()])
            nodes_df.to_csv(f"{base_name}_nodes.csv", index=False)
            
            edges_df = pd.DataFrame([edge.to_dict() for edge in self.edges])
            edges_df.to_csv(f"{base_name}_edges.csv", index=False)
            
            print(f"📊 Exported nodes to: {base_name}_nodes.csv")
            print(f"📊 Exported edges to: {base_name}_edges.csv")
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def visualize_graph(self, output_path: str = 'repo_graph.html', max_nodes: int = 100):
        """Create interactive visualization using Plotly"""
        print(f"🎨 Creating interactive visualization...")
        
        # Limit nodes for visualization performance
        important_nodes = sorted(
            self.nodes.values(), 
            key=lambda x: x.importance_score, 
            reverse=True
        )[:max_nodes]
        
        important_node_ids = {node.id for node in important_nodes}
        
        # Filter edges to only include important nodes
        filtered_edges = [
            edge for edge in self.edges 
            if edge.source in important_node_ids and edge.target in important_node_ids
        ]
        
        # Create subgraph
        subgraph = nx.DiGraph()
        for node in important_nodes:
            subgraph.add_node(node.id, **node.to_dict())
        
        for edge in filtered_edges:
            subgraph.add_edge(edge.source, edge.target, **edge.to_dict())
        
        # Layout
        try:
            pos = nx.spring_layout(subgraph, k=3, iterations=50)
        except:
            pos = {node: (i % 10, i // 10) for i, node in enumerate(subgraph.nodes())}
        
        # Prepare node data
        node_x = [pos[node][0] for node in subgraph.nodes()]
        node_y = [pos[node][1] for node in subgraph.nodes()]
        
        node_info = []
        node_colors = []
        node_sizes = []
        
        color_map = {
            'file': 'lightblue',
            'function_definition': 'lightgreen',
            'class_definition': 'orange',
            'method_definition': 'yellow',
        }
        
        for node_id in subgraph.nodes():
            node = self.nodes[node_id]
            node_info.append(
                f"<b>{node.name}</b><br>"
                f"Type: {node.node_type}<br>"
                f"File: {node.file_path}<br>"
                f"Language: {node.language}<br>"
                f"Tokens: {node.token_count}<br>"
                f"Complexity: {node.complexity_score:.2f}<br>"
                f"Importance: {node.importance_score:.2f}"
            )
            node_colors.append(color_map.get(node.node_type, 'gray'))
            node_sizes.append(max(10, min(30, node.importance_score * 20)))
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Dependencies'
        )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node.name for node in important_nodes],
            textposition="middle center",
            hovertext=node_info,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='black')
            ),
            name='Code Elements'
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Repository Dependency Graph (Top {len(important_nodes)} nodes)',
                        #    titlefont_size=16,
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Interactive repository dependency visualization",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='#888', size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        fig.write_html(output_path)
        print(f"📊 Interactive visualization saved to: {output_path}")

    def get_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Provide recommendations based on graph analysis"""
        recommendations = []
        
        # Check for circular dependencies
        circular_deps = analysis['clusters'].get('circular_dependencies', [])
        if circular_deps:
            recommendations.append(
                f"🔄 Found {len(circular_deps)} circular dependency cycles. "
                "Consider refactoring to break these cycles."
            )
        
        # Check graph density
        density = analysis['basic_stats']['density']
        if density > 0.3:
            recommendations.append(
                f"🕸️ High dependency density ({density:.2f}). "
                "Consider modularizing to reduce coupling."
            )
        elif density < 0.05:
            recommendations.append(
                f"🏝️ Low dependency density ({density:.2f}). "
                "Files might be too isolated - consider better integration."
            )
        
        # Check for highly central nodes
        top_central = analysis['centrality'].get('top_central_nodes', [])
        if top_central and top_central[0]['score'] > 0.5:
            recommendations.append(
                f"🎯 {top_central[0]['name']} has very high centrality ({top_central[0]['score']:.2f}). "
                "This file is critical - ensure it's well-tested and documented."
            )
        
        # Language diversity
        languages = analysis['basic_stats']['languages']
        if len(languages) > 5:
            recommendations.append(
                f"🌍 Repository uses {len(languages)} languages. "
                "Consider consolidating to reduce complexity."
            )
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description='Build dependency graph from parsed repository data')
    parser.add_argument('input_path', help='Path to parsed repository data (parquet/json/csv)')
    parser.add_argument('--output', '-o', default='repo_graph', help='Output file base name (default: repo_graph)')
    parser.add_argument('--format', choices=['gexf', 'json', 'graphml', 'csv'], default='json', 
                       help='Output format (default: json)')
    parser.add_argument('--visualize', action='store_true', help='Create interactive visualization')
    parser.add_argument('--analysis', action='store_true', help='Perform detailed graph analysis')
    parser.add_argument('--max-nodes', type=int, default=100, help='Maximum nodes for visualization (default: 100)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.input_path):
        print(f"❌ Error: Input file '{args.input_path}' does not exist")
        return 1
    
    try:
        # Initialize graph builder
        builder = RepoGraphBuilder(verbose=args.verbose)
        
        # Load parsed data
        df = builder.load_parsed_data(args.input_path)
        
        # Build graph
        graph = builder.build_graph(df)
        
        # Export graph
        output_path = f"{args.output}.{args.format}"
        builder.export_graph(output_path, args.format)

        print("\n Preparing graph data for optimized storage...")
        graph_data = {
            'nodes' : [node.to_dict() for node in builder.nodes.values()],
            'edges' : [edge.to_dict() for edge in builder.edges]
        }

        storage_path = f"{args.output}_storage"
        storage = OptimizedGraphStorage(storage_dir=storage_path)
        storage.save_graph(graph_data)

        print(f"\n✅ Graph processing and storage complete!")
        print(f"📦 Optimized storage created at: {storage_path}")

        
        # Perform analysis
        if args.analysis:
            print("\n🔍 Performing graph analysis...")
            analysis = builder.analyze_graph()
            
            # Print summary
            print("\n" + "="*60)
            print("📈 REPOSITORY GRAPH ANALYSIS")
            print("="*60)
            
            stats = analysis['basic_stats']
            print(f"📊 Graph Statistics:")
            print(f"  Nodes: {stats['nodes']:,}")
            print(f"  Edges: {stats['edges']:,}")
            print(f"  Density: {stats['density']:.3f}")
            print(f"  Connected: {'Yes' if stats['is_connected'] else 'No'}")
            
            print(f"\n🔤 Languages: {', '.join(stats['languages'].keys())}")
            
            print(f"\n🎯 Most Important Files:")
            for item in analysis['complexity']['most_important_nodes'][:5]:
                print(f"  • {item['name']} ({item['file']}) - Score: {item['score']:.2f}")
            
            print(f"\n🔗 Most Central Files:")
            for item in analysis['centrality']['top_central_nodes'][:5]:
                print(f"  • {item['name']} ({item['file']}) - Score: {item['score']:.2f}")
            
            # Get recommendations
            recommendations = builder.get_recommendations(analysis)
            if recommendations:
                print(f"\n💡 Recommendations:")
                for rec in recommendations:
                    print(f"  {rec}")
                    
            # Save detailed analysis
            analysis_path = f"{args.output}_analysis.json"
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"\n📋 Detailed analysis saved to: {analysis_path}")
        
        # Create visualization
        if args.visualize:
            viz_path = f"{args.output}_visualization.html"
            builder.visualize_graph(viz_path, args.max_nodes)
        
        print(f"\n✅ Graph processing complete!")
        print(f"📊 Graph exported to: {output_path}")
        
        if args.visualize:
            print(f"🎨 Interactive visualization saved to: {viz_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())