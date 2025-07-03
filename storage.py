import json
import sqlite3
import gzip
import pickle
import networkx as nx
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib
import time


class OptimizedGraphStorage:
    """Optimized storage system for repository dependency graphs"""
    
    def __init__(self, storage_dir: str = "graph_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # File paths
        self.json_path = self.storage_dir / "graph.json.gz"
        self.sqlite_path = self.storage_dir / "graph.db"
        self.cache_path = self.storage_dir / "graph_cache.pkl"
        self.metadata_path = self.storage_dir / "metadata.json"
        
        # In-memory caches
        self.node_cache = {}
        self.edge_cache = {}
        self.query_cache = {}
        
    def save_graph(self, graph_data: Dict[str, Any], force_rebuild: bool = False):
        """Save graph in multiple optimized formats"""
        
        # Generate content hash for change detection
        content_hash = self._generate_hash(graph_data)
        metadata = self._load_metadata()
        
        if not force_rebuild and metadata.get('hash') == content_hash:
            print("📊 Graph unchanged, skipping save")
            return
        
        print("💾 Saving graph in multiple formats...")
        
        # 1. Compressed JSON (primary format for LLM)
        self._save_compressed_json(graph_data)
        
        # 2. SQLite (for complex queries)
        self._save_to_sqlite(graph_data)
        
        # 3. NetworkX cache (for graph operations)
        self._save_networkx_cache(graph_data)
        
        # 4. Update metadata
        metadata = {
            'hash': content_hash,
            'timestamp': time.time(),
            'node_count': len(graph_data.get('nodes', [])),
            'edge_count': len(graph_data.get('edges', [])),
            'languages': list(set(node.get('language', 'unknown') 
                                for node in graph_data.get('nodes', []))),
            'version': '1.0'
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Graph saved successfully")
        print(f"   📄 Compressed JSON: {self.json_path}")
        print(f"   🗄️ SQLite DB: {self.sqlite_path}")
        print(f"   🔄 NetworkX cache: {self.cache_path}")
    
    def _save_compressed_json(self, graph_data: Dict[str, Any]):
        """Save as compressed JSON"""
        # Optimize JSON structure for LLM consumption
        optimized_data = {
            'metadata': {
                'node_count': len(graph_data.get('nodes', [])),
                'edge_count': len(graph_data.get('edges', [])),
                'languages': list(set(node.get('language', 'unknown') 
                                    for node in graph_data.get('nodes', []))),
                'timestamp': time.time()
            },
            'nodes': self._optimize_nodes_for_llm(graph_data.get('nodes', [])),
            'edges': self._optimize_edges_for_llm(graph_data.get('edges', [])),
            'summary': self._generate_graph_summary(graph_data)
        }
        
        # Compress and save
        json_str = json.dumps(optimized_data, separators=(',', ':'))
        with gzip.open(self.json_path, 'wt', encoding='utf-8') as f:
            f.write(json_str)
    
    def _save_to_sqlite(self, graph_data: Dict[str, Any]):
        """Save to SQLite for fast queries"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # Create optimized tables
        cursor.executescript('''
            DROP TABLE IF EXISTS nodes;
            DROP TABLE IF EXISTS edges;
            DROP TABLE IF EXISTS node_search;
            
            CREATE TABLE nodes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                language TEXT,
                importance_score REAL DEFAULT 0,
                complexity_score REAL DEFAULT 0,
                centrality_score REAL DEFAULT 0,
                token_count INTEGER DEFAULT 0,
                symbol_count INTEGER DEFAULT 0
            );
            
            CREATE TABLE edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY (source) REFERENCES nodes(id),
                FOREIGN KEY (target) REFERENCES nodes(id)
            );
            
            CREATE TABLE node_search (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                search_term TEXT NOT NULL,
                FOREIGN KEY (node_id) REFERENCES nodes(id)
            );
            
            -- Indexes for fast retrieval
            CREATE INDEX idx_nodes_type ON nodes(type);
            CREATE INDEX idx_nodes_importance ON nodes(importance_score DESC);
            CREATE INDEX idx_nodes_language ON nodes(language);
            CREATE INDEX idx_nodes_file ON nodes(file_path);
            CREATE INDEX idx_edges_source ON edges(source);
            CREATE INDEX idx_edges_target ON edges(target);
            CREATE INDEX idx_edges_type ON edges(edge_type);
            CREATE INDEX idx_search_term ON node_search(search_term);
        ''')
        
        # Insert nodes
        for node in graph_data.get('nodes', []):
            cursor.execute('''
                INSERT INTO nodes (id, name, type, file_path, language, 
                                 importance_score, complexity_score, centrality_score,
                                 token_count, symbol_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                node['id'],
                node.get('name', ''),
                node.get('node_type', 'unknown'),
                node.get('file_path', ''),
                node.get('language', 'unknown'),
                node.get('importance_score', 0),
                node.get('complexity_score', 0),
                node.get('centrality_score', 0),
                node.get('token_count', 0),
                node.get('symbol_count', 0)
            ))
            
            # Add search terms
            search_terms = self._extract_search_terms(node)
            for term in search_terms:
                cursor.execute('''
                    INSERT INTO node_search (node_id, search_term)
                    VALUES (?, ?)
                ''', (node['id'], term.lower()))
        
        # Insert edges
        for edge in graph_data.get('edges', []):
            cursor.execute('''
                INSERT INTO edges (source, target, edge_type, weight, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                edge['source'],
                edge['target'],
                edge.get('edge_type', 'unknown'),
                edge.get('weight', 1.0),
                edge.get('confidence', 1.0)
            ))
        
        conn.commit()
        conn.close()
    
    def _save_networkx_cache(self, graph_data: Dict[str, Any]):
        """Save NetworkX graph for fast graph operations"""
        G = nx.DiGraph()
        
        # Add nodes
        for node in graph_data.get('nodes', []):
            G.add_node(node['id'], **node)
        
        # Add edges
        for edge in graph_data.get('edges', []):
            G.add_edge(edge['source'], edge['target'], **edge)
        
        # Save as pickle for fast loading
        with open(self.cache_path, 'wb') as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_graph(self, format_type: str = "json") -> Any:
        """Load graph in specified format"""
        
        if format_type == "json":
            return self._load_compressed_json()
        elif format_type == "sqlite":
            return sqlite3.connect(self.sqlite_path)
        elif format_type == "networkx":
            return self._load_networkx_cache()
        else:
            raise ValueError(f"Unknown format: {format_type}")
    
    def _load_compressed_json(self) -> Dict[str, Any]:
        """Load compressed JSON"""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Graph file not found: {self.json_path}")
        
        with gzip.open(self.json_path, 'rt', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_networkx_cache(self) -> nx.DiGraph:
        """Load NetworkX graph from cache"""
        if not self.cache_path.exists():
            raise FileNotFoundError(f"NetworkX cache not found: {self.cache_path}")
        
        with open(self.cache_path, 'rb') as f:
            return pickle.load(f)
    
    def query_nodes(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Fast node search using SQLite"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # Search by multiple criteria
        search_query = '''
            SELECT DISTINCT n.* FROM nodes n
            LEFT JOIN node_search ns ON n.id = ns.node_id
            WHERE 
                n.name LIKE ? OR
                n.file_path LIKE ? OR
                ns.search_term LIKE ?
            ORDER BY n.importance_score DESC
            LIMIT ?
        '''
        
        search_term = f"%{query.lower()}%"
        cursor.execute(search_query, (search_term, search_term, search_term, limit))
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        conn.close()
        return results
    
    def get_node_dependencies(self, node_id: str, depth: int = 2) -> List[Dict[str, Any]]:
        """Get node dependencies using SQL"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # Recursive CTE for dependency traversal
        dependency_query = '''
            WITH RECURSIVE dependencies(node_id, depth) AS (
                SELECT ? as node_id, 0 as depth
                UNION ALL
                SELECT e.target, d.depth + 1
                FROM dependencies d
                JOIN edges e ON d.node_id = e.source
                WHERE d.depth < ?
            )
            SELECT DISTINCT n.* FROM nodes n
            JOIN dependencies d ON n.id = d.node_id
            ORDER BY d.depth, n.importance_score DESC
        '''
        
        cursor.execute(dependency_query, (node_id, depth))
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        conn.close()
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        metadata = self._load_metadata()
        
        if self.sqlite_path.exists():
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            # Get detailed stats
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_nodes,
                    COUNT(DISTINCT type) as node_types,
                    COUNT(DISTINCT language) as languages,
                    AVG(importance_score) as avg_importance,
                    MAX(importance_score) as max_importance
                FROM nodes
            ''')
            
            stats = cursor.fetchone()
            
            cursor.execute('SELECT COUNT(*) FROM edges')
            edge_count = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT type, COUNT(*) as count 
                FROM nodes 
                GROUP BY type 
                ORDER BY count DESC
            ''')
            
            node_type_dist = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'total_nodes': stats[0],
                'total_edges': edge_count,
                'node_types': stats[1],
                'languages': stats[2],
                'avg_importance': round(stats[3] or 0, 3),
                'max_importance': round(stats[4] or 0, 3),
                'node_type_distribution': node_type_dist,
                'last_updated': metadata.get('timestamp', 0),
                'storage_size': {
                    'json_gz': self.json_path.stat().st_size if self.json_path.exists() else 0,
                    'sqlite': self.sqlite_path.stat().st_size if self.sqlite_path.exists() else 0,
                    'cache': self.cache_path.stat().st_size if self.cache_path.exists() else 0
                }
            }
        
        return metadata
    
    def _optimize_nodes_for_llm(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize node structure for LLM consumption"""
        optimized = []
        
        for node in nodes:
            # Keep only essential fields for LLM
            optimized_node = {
                'id': node['id'],
                'name': node.get('name', ''),
                'type': node.get('node_type', 'unknown'),
                'file': node.get('file_path', ''),
                'lang': node.get('language', ''),
                'importance': round(node.get('importance_score', 0), 2),
                'complexity': round(node.get('complexity_score', 0), 2),
                'centrality': round(node.get('centrality_score', 0), 2),
                'tokens': node.get('token_count', 0),
                'symbols': node.get('symbol_count', 0)
            }

            optimized_node = {k: v for k, v in optimized_node.items() 
                            if v not in [0, 0.0, '', None]}
            
            optimized.append(optimized_node)
        
        return optimized
    

    def _optimize_edges_for_llm(self, edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize edge structure for LLM consumption"""
        optimized = []
        
        for edge in edges:
            optimized_edge = {
                'from': edge['source'],
                'to': edge['target'],
                'type': edge.get('edge_type', 'unknown'),
                'weight': round(edge.get('weight', 1.0), 2),
                'confidence': round(edge.get('confidence', 1.0), 2)
            }
            
            # Only include non-default values
            if optimized_edge['weight'] == 1.0:
                del optimized_edge['weight']
            if optimized_edge['confidence'] == 1.0:
                del optimized_edge['confidence']
            
            optimized.append(optimized_edge)
        
        return optimized
    
    def _generate_graph_summary(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the graph for LLM context"""
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        
        # Language distribution
        languages = {}
        for node in nodes:
            lang = node.get('language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1
        
        # Node type distribution
        node_types = {}
        for node in nodes:
            node_type = node.get('node_type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Edge type distribution
        edge_types = {}
        for edge in edges:
            edge_type = edge.get('edge_type', 'unknown')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        # Most important nodes
        important_nodes = sorted(
            nodes, 
            key=lambda x: x.get('importance_score', 0), 
            reverse=True
        )[:10]
        
        # Most central nodes
        central_nodes = sorted(
            nodes,
            key=lambda x: x.get('centrality_score', 0),
            reverse=True
        )[:10]
        
        return {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'languages': languages,
            'node_types': node_types,
            'edge_types': edge_types,
            'most_important': [
                {
                    'name': node.get('name', ''),
                    'file': node.get('file_path', ''),
                    'score': round(node.get('importance_score', 0), 2)
                }
                for node in important_nodes
            ],
            'most_central': [
                {
                    'name': node.get('name', ''),
                    'file': node.get('file_path', ''),
                    'score': round(node.get('centrality_score', 0), 2)
                }
                for node in central_nodes
            ]
        }
    
    def _extract_search_terms(self, node: Dict[str, Any]) -> List[str]:
        """Extract searchable terms from a node"""
        terms = []
        
        # Add name and variations
        name = node.get('name', '')
        if name:
            terms.append(name)
            # Add camelCase splits
            import re
            camel_splits = re.sub('([a-z0-9])([A-Z])', r'\1 \2', name).split()
            terms.extend(camel_splits)
            # Add snake_case splits
            snake_splits = name.replace('_', ' ').split()
            terms.extend(snake_splits)
        
        # Add file path components
        file_path = node.get('file_path', '')
        if file_path:
            path_parts = file_path.replace('/', ' ').replace('\\', ' ').split()
            terms.extend(path_parts)
        
        # Add node type
        node_type = node.get('node_type', '')
        if node_type:
            terms.append(node_type)
        
        # Add language
        language = node.get('language', '')
        if language:
            terms.append(language)
        
        # Clean and deduplicate
        cleaned_terms = []
        for term in terms:
            term = term.strip().lower()
            if term and len(term) > 1 and term not in cleaned_terms:
                cleaned_terms.append(term)
        
        return cleaned_terms
    
    def _generate_hash(self, data: Dict[str, Any]) -> str:
        """Generate hash for change detection"""
        # Create a stable string representation
        stable_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.md5(stable_str.encode()).hexdigest()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def get_file_dependencies(self, file_path: str) -> Dict[str, Any]:
        """Get all dependencies for a specific file"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # Get file node
        cursor.execute('SELECT * FROM nodes WHERE file_path = ? AND type = "file"', (file_path,))
        file_node = cursor.fetchone()
        
        if not file_node:
            conn.close()
            return {'error': 'File not found'}
        
        file_id = file_node[0]  # node id
        
        # Get direct dependencies (outgoing edges)
        cursor.execute('''
            SELECT n.name, n.file_path, e.edge_type, e.weight
            FROM edges e
            JOIN nodes n ON e.target = n.id
            WHERE e.source = ?
            ORDER BY e.weight DESC
        ''', (file_id,))
        
        dependencies = []
        for row in cursor.fetchall():
            dependencies.append({
                'name': row[0],
                'file_path': row[1],
                'edge_type': row[2],
                'weight': row[3]
            })
        
        # Get dependents (incoming edges)
        cursor.execute('''
            SELECT n.name, n.file_path, e.edge_type, e.weight
            FROM edges e
            JOIN nodes n ON e.source = n.id
            WHERE e.target = ?
            ORDER BY e.weight DESC
        ''', (file_id,))
        
        dependents = []
        for row in cursor.fetchall():
            dependents.append({
                'name': row[0],
                'file_path': row[1],
                'edge_type': row[2],
                'weight': row[3]
            })
        
        conn.close()
        
        return {
            'file_path': file_path,
            'dependencies': dependencies,
            'dependents': dependents,
            'dependency_count': len(dependencies),
            'dependent_count': len(dependents)
        }
    
    def find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in the graph"""
        try:
            G = self._load_networkx_cache()
            cycles = list(nx.simple_cycles(G))
            
            # Convert node IDs to readable names
            readable_cycles = []
            for cycle in cycles:
                readable_cycle = []
                for node_id in cycle:
                    if node_id in G.nodes:
                        node_data = G.nodes[node_id]
                        readable_cycle.append({
                            'name': node_data.get('name', ''),
                            'file': node_data.get('file_path', ''),
                            'type': node_data.get('node_type', '')
                        })
                readable_cycles.append(readable_cycle)
            
            return readable_cycles
        except:
            return []
        
    def get_critical_paths(self, start_node: str, end_node: str) -> List[List[str]]:
        """Find critical paths between two nodes"""
        try:
            G = self._load_networkx_cache()
            
            if start_node not in G.nodes or end_node not in G.nodes:
                return []
            
            # Find all simple paths
            paths = list(nx.all_simple_paths(G, start_node, end_node, cutoff=6))
            
            # Convert to readable format
            readable_paths = []
            for path in paths[:5]:  # Limit to first 5 paths
                readable_path = []
                for node_id in path:
                    if node_id in G.nodes:
                        node_data = G.nodes[node_id]
                        readable_path.append({
                            'name': node_data.get('name', ''),
                            'file': node_data.get('file_path', ''),
                            'importance': node_data.get('importance_score', 0)
                        })
                readable_paths.append(readable_path)
            
            return readable_paths
        except:
            return []
        
    def cleanup_storage(self):
        """Clean up storage files"""
        files_to_remove = [
            self.json_path,
            self.sqlite_path,
            self.cache_path,
            self.metadata_path
        ]
        
        for file_path in files_to_remove:
            if file_path.exists():
                file_path.unlink()
        
        # Remove directory if empty
        try:
            self.storage_dir.rmdir()
        except OSError:
            pass  # Directory not empty

    def export_for_llm(self, output_path: str, max_nodes: int = 500):
        """Export a simplified version optimized for LLM analysis"""
        try:
            graph_data = self._load_compressed_json()
            
            # Limit to most important nodes
            nodes = graph_data.get('nodes', [])
            important_nodes = sorted(
                nodes,
                key=lambda x: x.get('importance', 0),
                reverse=True
            )[:max_nodes]
            
            important_node_ids = {node['id'] for node in important_nodes}
            
            # Filter edges to only include important nodes
            edges = graph_data.get('edges', [])
            filtered_edges = [
                edge for edge in edges
                if edge['from'] in important_node_ids and edge['to'] in important_node_ids
            ]
            
            # Create simplified export
            llm_export = {
                'summary': graph_data.get('summary', {}),
                'nodes': important_nodes,
                'edges': filtered_edges,
                'analysis': {
                    'total_original_nodes': len(nodes),
                    'exported_nodes': len(important_nodes),
                    'exported_edges': len(filtered_edges),
                    'compression_ratio': len(important_nodes) / len(nodes) if nodes else 0
                }
            }
            
            # Save as compressed JSON
            with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                json.dump(llm_export, f, indent=2)
            
            print(f"📤 LLM-optimized graph exported to: {output_path}")
            print(f"   📊 Nodes: {len(important_nodes)}/{len(nodes)}")
            print(f"   🔗 Edges: {len(filtered_edges)}/{len(edges)}")
            
        except Exception as e:
            print(f"❌ Error exporting for LLM: {e}")

if __name__ == "__main__":
    # Initialize storage
    storage = OptimizedGraphStorage("my_repo_graph_storage")
    
    # Example usage with graph data from the main graph.py
    # This would typically be called from the main graph building script
    
    # Load existing graph if it exists
    try:
        stats = storage.get_statistics()
        print("📊 Current graph statistics:")
        print(f"   Nodes: {stats.get('total_nodes', 0)}")
        print(f"   Edges: {stats.get('total_edges', 0)}")
        print(f"   Languages: {stats.get('languages', 0)}")
        
        # Example queries
        print("\n🔍 Example queries:")
        
        # Search for nodes
        results = storage.query_nodes("main", limit=5)
        print(f"   Found {len(results)} nodes matching 'main'")

        if not results:
            print("NO Results found :(())")
        
        for i, node in enumerate(results):
            print(f"  {i + 1} Name  : {node.get('name','N/A')}")
            print(f"         Type: {node.get('type', 'N/A')}")
            print(f"         File: {node.get('file_path', 'N/A')}")
            print(f"         Importance: {node.get('importance_score', 0):.2f}")
        
        # Find circular dependencies
        cycles = storage.find_circular_dependencies()
        print(f"   Found {len(cycles)} circular dependencies")
        
        # Export for LLM
        storage.export_for_llm("graph_for_llm.json.gz", max_nodes=100)
        
    except FileNotFoundError:
        print("📝 No existing graph found. Run the main graph builder first.")
    except Exception as e:
        print(f"❌ Error: {e}")