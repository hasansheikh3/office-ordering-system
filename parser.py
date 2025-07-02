import os
import uuid
import json
import argparse
import tiktoken
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pathspec
from typing import List, Dict, Any, Generator, Optional
from collections import Counter
from tree_sitter import Language, Parser, Node
from tqdm import tqdm
import time
import datetime
import concurrent.futures
import threading

# Import tree-sitter language modules
try:
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjavascript
    import tree_sitter_typescript as tstypescript
    import tree_sitter_go as tsgo
    import tree_sitter_c as tsc
    import tree_sitter_cpp as tscpp
    import tree_sitter_rust as tsrust
    import tree_sitter_java as tsjava
    import tree_sitter_ruby as tsruby
    import tree_sitter_php as tsphp
    import tree_sitter_html as tshtml
    import tree_sitter_css as tscss
    import tree_sitter_bash as tsbash
except ImportError as e:
    print(f"Warning: Some tree-sitter language modules not available: {e}")
    print("Install with: pip install tree-sitter-python tree-sitter-javascript etc.")


class RepoParser:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, ignore_dirs: List[str] = None, ignore_files: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.ignore_dirs = ignore_dirs if ignore_dirs else []
        self.ignore_files = ignore_files if ignore_files else []
        self.language_map = {
            ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
            ".go": "go", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp",
            ".rs": "rust", ".java": "java", ".kt": "kotlin", ".swift": "swift", ".rb": "ruby", ".php": "php",
            ".md": "markdown", ".rst": "rst", ".txt": "text", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
            ".toml": "toml", ".ini": "ini", ".cfg": "ini", ".conf": "ini",
            # Add these missing extensions:
            ".html": "html", ".htm": "html", ".css": "css", ".scss": "scss", ".sass": "scss",
            ".xml": "xml", ".svg": "xml", ".sql": "sql", ".sh": "bash", ".bash": "bash",
            ".ps1": "powershell", ".bat": "batch", ".dockerfile": "dockerfile", ".gitignore": "text",
            ".cs" : "csharp", ".kt" : "kotlin", ".dart" : "dart", ".scala" : "scala", ".hs" : "haskell",
            ".lua" : "lua", ".properties" : "properties"
        }
        self.ts_languages = {}
        self._language_lock = threading.Lock()
        self.language_modules = {
            'python': tspython if 'tspython' in globals() else None,
            'javascript': tsjavascript if 'tsjavascript' in globals() else None,
            'typescript': tstypescript if 'tstypescript' in globals() else None,
            'go': tsgo if 'tsgo' in globals() else None,
            'c': tsc if 'tsc' in globals() else None,
            'cpp': tscpp if 'tscpp' in globals() else None,
            'rust': tsrust if 'tsrust' in globals() else None,
            'java': tsjava if 'tsjava' in globals() else None,
            'ruby': tsruby if 'tsruby' in globals() else None,
            'php': tsphp if 'tsphp' in globals() else None,
            'html': tshtml if 'tshtml' in globals() else None,
            'css': tscss if 'tscss' in globals() else None,
            'bash': tsbash if 'tsbash' in globals() else None,
        }

        
        # Enhanced Tree-sitter queries for better parsing
        self.queries = {
            "python": {
                "imports": """
                    (import_statement) @import
                    (import_from_statement) @import
                """,
                "out_calls": "(call) @call",
                "symbols": ["function_definition", "class_definition", "assignment"],
                "docstring": """
                    (function_definition
                        body: (block
                            (expression_statement
                                (string) @docstring)))
                    (class_definition
                        body: (block
                            (expression_statement
                                (string) @docstring)))
                """,
                "variables": "(assignment left: (identifier) @var_name) @assignment",
                "decorators": "(decorator) @decorator"
            },
            "javascript": {
                "imports": """
                    (import_statement) @import
                    (import_clause) @import
                """,
                "out_calls": "(call_expression) @call",
                "symbols": ["function_declaration", "class_declaration", "method_definition", "variable_declaration"],
                "variables": "(variable_declaration) @var_decl"
            },
            "typescript": {
                "imports": """
                    (import_statement) @import
                    (import_clause) @import
                """,
                "out_calls": "(call_expression) @call",
                "symbols": ["function_declaration", "class_declaration", "method_definition", "interface_declaration", "type_alias_declaration", "variable_declaration"],
                "variables": "(variable_declaration) @var_decl"
            },
            "go": {
                "imports": "(import_declaration) @import",
                "out_calls": "(call_expression) @call",
                "symbols": ["function_declaration", "method_declaration", "type_spec", "var_declaration", "const_declaration"],
                "variables": """
                    (var_declaration) @var_decl
                    (const_declaration) @const_decl
                """
            },
            "c": {
                "imports": "(preproc_include) @import",
                "out_calls": "(call_expression) @call",
                "symbols": ["function_definition", "struct_specifier", "enum_specifier", "declaration"],
                "variables": "(declaration) @var_decl"
            },
            "cpp": {
                "imports": """
                    (preproc_include) @import
                    (using_declaration) @import
                    (namespace_definition) @namespace
                """,
                "out_calls": "(call_expression) @call",
                "symbols": ["function_definition", "class_specifier", "struct_specifier", "enum_specifier", "declaration", "namespace_definition"],
                "variables": "(declaration) @var_decl"
            },
            "rust": {
                "imports": """
                    (use_declaration) @import
                    (extern_crate_declaration) @import
                """,
                "out_calls": "(call_expression) @call",
                "symbols": ["function_item", "struct_item", "enum_item", "impl_item", "trait_item", "mod_item", "const_item", "static_item"],
                "variables": """
                    (const_item) @const
                    (static_item) @static
                    (let_declaration) @let_decl
                """
            },
            "java": {
                "imports": """
                    (import_declaration) @import
                    (package_declaration) @import
                """,
                "out_calls": "(method_invocation) @call",
                "symbols": ["class_declaration", "interface_declaration", "method_declaration", "field_declaration"],
                "variables": "(field_declaration) @field"
            }, 
            "csharp": {
                "imports": "(using_directive) @import",
                "out_calls": "(invocation_expression) @call",
                "symbols": ["class_declaration", "method_declaration", "struct_declaration"],
                "function_signature": "(method_declaration name: (identifier) @func_name parameters: (parameter_list) @params)",
                "class_signature": "(class_declaration name: (identifier) @class_name)"
            },
            "kotlin": {
                "imports": "(import_header) @import",
                "out_calls": "(call_expression) @call",
                "symbols": ["class_declaration", "function_declaration", "object_declaration"],
                "function_signature": "(function_declaration name: (simple_identifier) @func_name parameters: (function_value_parameters) @params)",
                "class_signature": "(class_declaration name: (type_identifier) @class_name)"
            }
        }

        # Enhanced queries to capture full signature
        self.queries["python"].update({
            "function_signature" : "(function_definition name: (identifier) @func_name parameters: (parameters) @params)",
            "class_signature": "(class_definition name: (identifier) @class_name)"
        })
        self.queries["javascript"].update({
            "function_signature": "(function_declaration name: (identifier) @func_name parameters: (formal_parameters) @params)",
            "class_signature": "(class_declaration name: (identifier) @class_name)"
        })
        self.queries["typescript"].update({
            "function_signature": "(function_declaration name: (identifier) @func_name parameters: (formal_parameters) @params)",
            "class_signature": "(class_declaration name: (identifier) @class_name)"
        })
        self.queries["go"].update({
            "function_signature": "(function_declaration name: (identifier) @func_name parameters: (parameter_list) @params)",
            "type_signature": "(type_spec name: (identifier) @type_name)"
        })
        self.queries["c"].update({
            "function_signature": "(function_definition declarator: (declarator identifier: (identifier) @func_name) parameters: (parameter_list) @params)",
            "struct_signature": "(struct_specifier name: (identifier) @struct_name)"
        })
        self.queries["cpp"].update({
            "function_signature": "(function_definition declarator: (declarator identifier: (identifier) @func_name) parameters: (parameter_list) @params)",
            "class_signature": "(class_specifier name: (identifier) @class_name)",
            "namespace_signature": "(namespace_definition name: (identifier) @namespace_name)"
        })
        self.queries["rust"].update({
            "function_signature": "(function_item name: (identifier) @func_name parameters: (parameter_list) @params)",
            "struct_signature": "(struct_item name: (identifier) @struct_name)",
            "enum_signature": "(enum_item name: (identifier) @enum_name)"
        })
        self.queries["java"].update({
            "function_signature": "(method_declaration name: (identifier) @func_name parameters: (formal_parameters) @params)",
            "class_signature": "(class_declaration name: (identifier) @class_name)",
            "interface_signature": "(interface_declaration name: (identifier) @interface_name)"
        })
        # Add queries for other languages as needed

    def _get_ignore_spec(self, repo_path: str) -> pathspec.PathSpec:
        patterns = []
        gitignore_path = os.path.join(repo_path, '.gitignore')
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r', encoding='utf-8', errors='ignore') as f:
                patterns.extend(f.readlines())
        
        # Add common ignore patterns
        patterns.extend([
            '*.pyc', '__pycache__/', '.git/', 'node_modules/', '.venv/', 'venv/',
            '*.log', '*.tmp', '.DS_Store', 'Thumbs.db', '*.swp', '*.swo',
            ".env", "*.secret", "*.key", "*.pem", "*.cert",
            "vendor/", "dist/", "build/", "out/", "target/"
        ])
        
        patterns.extend(self.ignore_dirs)
        patterns.extend(self.ignore_files)
        return pathspec.PathSpec.from_lines('gitwildmatch', patterns)

    def get_files_to_parse(self, repo_path: str) -> List[str]:
        spec = self._get_ignore_spec(repo_path)
        files_to_parse = []
        for root, dirs, files in os.walk(repo_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not spec.match_file(os.path.join(root, d))]
            
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                if not spec.match_file(relative_path) and self._is_parseable_file(file_path):
                    files_to_parse.append(relative_path)
        return sorted(files_to_parse)

    def _is_parseable_file(self, file_path: str) -> bool:
        """Check if file is parseable and not too large"""
        try:
            # Skip very large files (>10MB)
            if os.path.getsize(file_path) > 10 * 1024 * 1024:
                return False
            
            # Check if it's a supported file type
            extension = os.path.splitext(file_path)[1].lower()
            return extension in self.language_map
        except (OSError, IOError):
            return False

    def display_summary(self, repo_path: str):
        files_to_parse = self.get_files_to_parse(repo_path)
        total_files = len(files_to_parse)
        total_token_count = 0
        language_counts = Counter()
        file_sizes = []

        for file_path in files_to_parse:
            # Get extension directly instead of using _get_language
            extension = os.path.splitext(file_path)[1].lower()
            lang_name = self.language_map.get(extension)
            
            if lang_name:  # This will now work for all supported languages
                language_counts[lang_name] += 1
                try:
                    full_path = os.path.join(repo_path, file_path)
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                    token_count = len(self.encoding.encode(code))
                    total_token_count += token_count
                    file_sizes.append(token_count)
                except (IOError, UnicodeDecodeError) as e:
                    print(f"Could not read file for summary: {file_path}, error: {e}")
        
        # Rest of the method stays the same...
        avg_tokens = sum(file_sizes) / len(file_sizes) if file_sizes else 0
        estimated_chunks = sum(max(1, size // self.chunk_size) for size in file_sizes)
        
        print("=" * 50)
        print("📊 REPOSITORY ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"📁 Total Files: {total_files}")
        print(f"💻 Code Files: {sum(language_counts.values())}")
        print(f"🔢 Total Tokens: {total_token_count:,}")
        print(f"📊 Average Tokens/File: {avg_tokens:.0f}")
        print(f"🧩 Estimated Chunks: {estimated_chunks:,}")
        print(f"⚙️  Chunk Size: {self.chunk_size} tokens")
        print(f"🔄 Chunk Overlap: {self.chunk_overlap} tokens")
        
        print("\n📈 Language Distribution:")
        for lang, count in language_counts.most_common():
            percentage = (count / sum(language_counts.values())) * 100
            print(f"  {lang.upper():>12}: {count:>4} files ({percentage:>5.1f}%)")
        
        if file_sizes:
            print(f"\n📏 File Size Distribution:")
            print(f"  Smallest: {min(file_sizes):,} tokens")
            print(f"  Largest:  {max(file_sizes):,} tokens")
            print(f"  Median:   {sorted(file_sizes)[len(file_sizes)//2]:,} tokens")
        
        print("=" * 50)

    def parse_repo(self, repo_path: str, max_workers: int = None) -> List[Dict[str, Any]]:
        """Parse repository with parallel processing for blazing speed"""
        files_to_parse = self.get_files_to_parse(repo_path)
        all_chunks = []
        
        # Determine optimal number of workers
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) + 4)  # Default from ThreadPoolExecutor
        
        print(f"🚀 Parsing {len(files_to_parse)} files with {max_workers} workers...")
        
        # Batch files for better progress tracking
        batch_size = max(1, len(files_to_parse) // 20)  # 20 progress updates
        successful_files = 0
        failed_files = 0
        total_chunks = 0
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all parsing tasks
            future_to_file = {
                executor.submit(self._parse_file_safe, os.path.join(repo_path, file_path), file_path, repo_path): file_path
                for file_path in files_to_parse
            }
            
            # Process completed tasks with progress tracking
            with tqdm(total=len(files_to_parse), desc="Parsing files", unit="files") as pbar:
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    
                    try:
                        file_chunks = future.result(timeout=30)  # 30s timeout per file
                        if file_chunks:
                            all_chunks.extend(file_chunks)
                            total_chunks += len(file_chunks)
                            successful_files += 1
                            pbar.set_postfix({
                                'chunks': total_chunks,
                                'success': successful_files,
                                'failed': failed_files
                            })
                        else:
                            failed_files += 1
                            
                    except concurrent.futures.TimeoutError:
                        print(f"  ⏰ Timeout parsing {file_path}")
                        failed_files += 1
                    except Exception as e:
                        print(f"  ❌ Error parsing {file_path}: {str(e)}")
                        failed_files += 1
                    
                    pbar.update(1)
        
        elapsed_time = time.time() - start_time
        
        # Enhanced completion stats
        print(f"\n✅ Parallel parsing complete!")
        print(f"⏱️  Total time: {elapsed_time:.2f}s")
        print(f"📊 Files processed: {successful_files}/{len(files_to_parse)} ({successful_files/len(files_to_parse)*100:.1f}%)")
        print(f"🧩 Total chunks: {total_chunks:,}")
        print(f"🚄 Average speed: {len(files_to_parse)/elapsed_time:.1f} files/sec")
        print(f"⚡ Chunk generation rate: {total_chunks/elapsed_time:.1f} chunks/sec")
        
        if failed_files > 0:
            print(f"⚠️  Failed files: {failed_files}")
        
        return all_chunks

    def _parse_file_safe(self, file_path: str, relative_path: str, repo_path: str) -> List[Dict[str, Any]]:
        """Thread-safe wrapper for file parsing with error handling"""
        try:
            # Skip very large files (>2MB) to prevent memory issues in parallel processing
            if os.path.getsize(file_path) > 2 * 1024 * 1024:
                print(f"  ⚠️ Skipping large file: {relative_path}")
                return []
            
            # Add parsing timeout for individual files
            start_time = time.time()
            file_chunks = self._parse_file(file_path, relative_path, repo_path)
            elapsed = time.time() - start_time
            
            # Log slow files for optimization
            if elapsed > 3:
                print(f"  🐌 Slow parse: {relative_path} took {elapsed:.2f}s")
            
            return file_chunks
            
        except Exception as e:
            # Detailed error logging for debugging
            print(f"  ❌ Parse error in {relative_path}: {type(e).__name__}: {str(e)}")
            return []
        
    def _get_language_thread_safe(self, file_path: str) -> Optional[Language]:
        """Thread-safe version of language detection"""
        extension = os.path.splitext(file_path)[1].lower()
        lang_name = self.language_map.get(extension)

        if not lang_name:
            return None

        # Handle document/config types (no tree-sitter needed)
        if lang_name in ["markdown", "rst", "text", "json", "yaml", "toml", "ini"]:
            class SimpleLang:
                def __init__(self, name): self.name = name
            return SimpleLang(lang_name)

        # Thread-safe language loading with lock
        with self._language_lock:
            # Return cached if already loaded
            if lang_name in self.ts_languages:
                return self.ts_languages[lang_name]

            try:
                ts_lang = Language(f"/home/hasan/.tree-sitter/repos/{lang_name}.so", lang_name)
                self.ts_languages[lang_name] = ts_lang
                return ts_lang
            except Exception as e:
                print(f"❌ Failed to load {lang_name}: {e}")
                return None


    def _parse_file(self, file_path: str, relative_path: str, repo_path: str) -> List[Dict[str, Any]]:
        language = self._get_language_thread_safe(file_path)
        if not language:
            return []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
                last_modified = os.path.getmtime(file_path)
        except (IOError, UnicodeDecodeError) as e:
            print(f"Error reading file {file_path}: {e}")
            return []
        
        # Handle LICENSE Files as single chunks
        base_name = os.path.basename(file_path).lower()
        if "license" in base_name or "copying" in base_name:
            token_count = len(self.encoding.encode(code))
            return [self._create_chunk_dict(
                relative_path, language.name, "license", code,
                0, len(code.splitlines()), token_count, {},
                last_modified=last_modified
            )]

        # Handle different file types
        if language.name in ["markdown", "rst", "text"]:
            return self._chunk_document(code, relative_path, language.name, last_modified)
        elif language.name in ["json", "yaml", "toml", "ini"]:
            return self._chunk_config(code, relative_path, language.name, last_modified)
        else:
            return self._parse_code_file(code, relative_path, language, repo_path, last_modified)

    def _get_module_path(self, relative_path: str, repo_path: str) -> str:
        """Get module path for any code file"""
        path_parts = relative_path.split(os.sep)
        filename = path_parts[-1]
        
        # Remove extension for code files
        code_extensions = {".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp"}
        if any(filename.endswith(ext) for ext in code_extensions):
            # Remove all extensions (handles cases like file.test.js)
            while "." in filename:
                filename = filename.rsplit(".", 1)[0]
            path_parts[-1] = filename
        
        return '.'.join(path_parts)

    

    def _parse_code_file(self, code: str, relative_path: str, language: Language, repo_path: str, last_modified : float) -> List[Dict[str, Any]]:
        parser = Parser()
        parser.set_language(language)
        tree = parser.parse(bytes(code, "utf8"))

        module_path = self._get_module_path(relative_path, repo_path)

        print(f"Parsing {relative_path} with Tree-sitter for {language.name}...")
        
        # Extract file-level context
        file_context = self._get_file_context(tree.root_node, language, code, relative_path, repo_path)
        file_context.update({
            "module" : module_path,
            "last_modified": last_modified,
        })
        
        chunks = []
        
        # Add file overview chunk
        file_overview = self._create_file_overview(code, relative_path, language, file_context)
        if file_overview:
            chunks.append(file_overview)
        
        print(f"Extracting symbols from {relative_path}...")
            
        # Parse symbols and create chunks (convert generator to list)
        chunks.extend(list(self._recursive_chunker(
            tree.root_node, code, relative_path, language, file_context
        )))

        print(f"Extracted {len(chunks)} chunks from {relative_path}")
        
        return chunks

    def _get_language(self, file_path: str) -> Optional[Language]:
        extension = os.path.splitext(file_path)[1].lower()
        lang_name = self.language_map.get(extension)

        if not lang_name:
            return None

        # Handle document/config types
        if lang_name in ["markdown", "rst", "text", "json", "yaml", "toml", "ini"]:
            class SimpleLang:
                def __init__(self, name): self.name = name
            return SimpleLang(lang_name)

        # Return cached if already loaded
        if lang_name in self.ts_languages:
            return self.ts_languages[lang_name]

        try:
            ts_lang = Language(f"/home/hasan/.tree-sitter/repos/{lang_name}.so", lang_name)
            self.ts_languages[lang_name] = ts_lang
            print(f"✅ Loaded Tree-sitter language: {lang_name}")
            return ts_lang
        except Exception as e:
            print(f"❌ Failed to load {lang_name} from tree-sitter: {e}")
            return None

    def _get_file_context(self, root_node: Node, language: Language, code: str, relative_path: str, repo_path: str) -> Dict[str, Any]:
        """Extract comprehensive file-level context"""
        context = {
            "imports": self._get_imports(root_node, language),
            "file_docstring": self._get_file_docstring(code, language),
            "exports": self._get_exports(root_node, language),
            "package_path": self._get_package_path(relative_path, repo_path),
            "file_size": len(code),
            "line_count": len(code.splitlines())
        }
        return context

    def _get_file_docstring(self, code: str, language: Language) -> Optional[str]:
        """Extract file-level docstring or comments"""
        lines = code.splitlines()
        if not lines:
            return None
            
        if language.name == "python":
            # Check for module docstring
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    # Find closing quotes
                    quote_type = stripped[:3]
                    if stripped.count(quote_type) >= 2:
                        return stripped.strip(quote_type).strip()
                    # Multi-line docstring
                    docstring_lines = [stripped[3:]]
                    for j in range(i + 1, len(lines)):
                        if quote_type in lines[j]:
                            docstring_lines.append(lines[j][:lines[j].find(quote_type)])
                            break
                        docstring_lines.append(lines[j])
                    return "\n".join(docstring_lines).strip()
                elif stripped and not stripped.startswith('#'):
                    break
        
        # Extract initial comment block for other languages
        comment_patterns = {
            "javascript": "//",
            "typescript": "//",
            "go": "//",
            "rust": "//",
            "c": "//",
            "cpp": "//",
            "java": "//"
        }
        
        comment_prefix = comment_patterns.get(language.name)
        if comment_prefix:
            comments = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(comment_prefix):
                    comments.append(stripped[len(comment_prefix):].strip())
                elif stripped and not stripped.startswith('#'):
                    break
            if comments:
                return "\n".join(comments)
        
        return None

    def _get_package_path(self, relative_path: str, repo_path: str) -> str:
        """Determine package/module path for the file"""
        path_parts = relative_path.split(os.sep)[:-1]  # Remove filename
        return ".".join(path_parts) if path_parts else ""

    def _get_exports(self, node: Node, language: Language) -> List[str]:
        """Extract exported symbols from the file"""
        exports = []
        if language.name == "javascript" or language.name == "typescript":
            # Look for export statements
            try:
                query = language.query("(export_statement) @export")
                captures = query.captures(node)
                for capture, _ in captures:
                    exports.append(capture.text.decode("utf-8"))
            except:
                pass
        return exports

    def _create_file_overview(self, code: str, file_path: str, language: Language, file_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a simple file overview"""
        lines = code.splitlines()
        if len(lines) < 10:  # Too small for overview
            return None
        
        # Get first 20 lines or until first function/class
        overview_lines = []
        for i, line in enumerate(lines[:20]):
            overview_lines.append(line)
            # Stop at first major symbol
            if any(keyword in line for keyword in ["def ", "class ", "function ", "interface "]):
                break
        
        overview_content = "\n".join(overview_lines)
        token_count = len(self.encoding.encode(overview_content))
        
        if token_count < 50:  # Too small to be useful
            return None
        
        return self._create_chunk_dict(
            file_path, language, "file_overview", overview_content,
            0, len(overview_lines), token_count, {}, is_overview=True
        )

    def _get_docstring(self, node: Node, language: Language) -> Optional[str]:
        """Extract docstring with improved accuracy"""
        if language.name not in self.queries or "docstring" not in self.queries[language.name]:
            return None
            
        try:
            query = language.query(self.queries[language.name]["docstring"])
            captures = query.captures(node)
            for capture, name in captures:
                if name == "docstring":
                    text = capture.text.decode("utf-8")
                    # Clean up the docstring
                    return text.strip('"""').strip("'''").strip()
        except Exception:
            pass
        
        return None

    def _get_imports(self, node: Node, language: Language) -> List[str]:
        imports = []
        if language.name in self.queries and "imports" in self.queries[language.name]:
            try:
                query = language.query(self.queries[language.name]["imports"])
                captures = query.captures(node)
                for capture, _ in captures:
                    import_text = capture.text.decode("utf-8").strip()
                    if import_text:
                        imports.append(import_text)
            except Exception as e:
                print(f"Error extracting imports: {e}")
        return list(set(imports))  # Remove duplicates


    def _get_out_calls(self, node: Node, language: Language) -> List[str]:
        out_calls = []
        if language.name in self.queries and "out_calls" in self.queries[language.name]:
            try:
                query = language.query(self.queries[language.name]["out_calls"])
                captures = query.captures(node)
                for capture, _ in captures:
                    call_text = capture.text.decode("utf-8").strip()
                    # Extract just the function name
                    if "(" in call_text:
                        call_name = call_text.split("(")[0].strip()
                        if call_name:
                            out_calls.append(call_name)
            except Exception as e:
                print(f"Error extracting out_calls: {e}")
        return list(set(out_calls))  # Remove duplicates
    
    def _split_chunk_simple(self, code: str, file_path: str, language: Language, symbol_type: str, 
                       start_line: int, context: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """Simple chunk splitting without overlap complexity"""
        lines = code.splitlines()
        
        # Split by logical sections (try to keep functions/classes together)
        if len(lines) <= 50:  # Small enough, don't split
            token_count = len(self.encoding.encode(code))
            yield self._create_chunk_dict(file_path, language, symbol_type, code, start_line, start_line + len(lines) - 1, token_count, context)
            return
        
        # Split into reasonable chunks
        chunk_size_lines = 30
        for i in range(0, len(lines), chunk_size_lines):
            chunk_lines = lines[i:i + chunk_size_lines]
            chunk_code = "\n".join(chunk_lines)
            
            if chunk_code.strip():
                token_count = len(self.encoding.encode(chunk_code))
                chunk_start_line = start_line + i
                chunk_end_line = start_line + i + len(chunk_lines) - 1
                
                yield self._create_chunk_dict(
                    file_path, language, f"split_{symbol_type}", chunk_code,
                    chunk_start_line, chunk_end_line, token_count, context
                )

    
    def _recursive_chunker(self, root_node: Node, code: str, file_path: str, language: Language, 
            file_context: Dict[str, Any], parent_context: Dict[str, Any] = None) -> Generator[Dict[str, Any], None, None]:
        """Simplified recursive chunker with function/class preservation"""
        
        # Skip tiny assignments and variable declarations - they're noise
        SKIP_SYMBOLS = {"assignment", "variable_declaration", "expression_statement"}
        
        # Only process meaningful symbols
        MEANINGFUL_SYMBOLS = {
            "function_definition", "class_definition", "method_definition",
            "function_declaration", "class_declaration", "interface_declaration",
            "struct_item", "impl_item", "trait_item", "enum_item"
        }
        
        stack = [(root_node, parent_context or {}, 0)]
        visited = set()
        processed = 0
        
        while stack and processed < 1000:  # Safety limit
            node, current_parent_context, depth = stack.pop()
            
            if id(node) in visited or depth > 20:
                continue
                
            visited.add(id(node))
            processed += 1
            
            # Only process meaningful symbols
            if (hasattr(node, 'type') and 
                node.type in MEANINGFUL_SYMBOLS and
                node.type not in SKIP_SYMBOLS):
                
                try:
                    start_line = node.start_point[0]
                    end_line = node.end_point[0]
                    symbol_code = "\n".join(code.splitlines()[start_line:end_line+1])
                    
                    if not symbol_code.strip():
                        continue
                        
                    token_count = len(self.encoding.encode(symbol_code))
                    symbol_name = self._get_symbol_name(node, language)
                    
                    # Skip if too small (likely noise)
                    if token_count < 10 and node.type not in ["class_definition", "class_declaration"]:
                        continue
                    
                    context = {
                        "symbol_name": symbol_name,
                        "parent_class": current_parent_context.get("class_name")
                    }
                    
                    # NEW: Only split if token count is 3x chunk size (keep functions/classes intact)
                    if token_count > self.chunk_size * 3:
                        # Split large chunks
                        yield from self._split_chunk_improved(symbol_code, file_path, language, node.type, start_line, context)
                    else:
                        # NEW: Get full symbol signature
                        symbol_signature = self._get_symbol_signature(node, language, code)
                        yield self._create_chunk_dict(
                            file_path, language, node.type, symbol_code, 
                            start_line, end_line, token_count, context,
                            symbol_signature=symbol_signature
                        )
                    
                    # Update parent context
                    new_parent_context = current_parent_context.copy()
                    if node.type in ["class_definition", "class_declaration"]:
                        new_parent_context["class_name"] = symbol_name
                        
                except Exception as e:
                    print(f"Error processing {node.type} in {file_path}: {e}")
                    continue
            else:
                new_parent_context = current_parent_context
            
            # Add children to stack
            if depth < 20 and hasattr(node, 'children'):
                for child in reversed(node.children):
                    if hasattr(child, 'type') and id(child) not in visited:
                        stack.append((child, new_parent_context, depth + 1))


    
    def _get_symbol_signature(self, node: Node, language: Language, code: str) -> str:
        """Get full function/class signature for any language"""
        if language.name == "python":
            if node.type == "function_definition":
                # Capture from 'def' to colon
                start = node.start_byte
                colon = code.find(":", node.start_byte)
                return code[start:colon + 1] if colon != -1 else node.text.decode("utf-8")
            elif node.type == "class_definition":
                # Capture from 'class' to colon
                start = node.start_byte
                colon = code.find(":", node.start_byte)
                return code[start:colon + 1] if colon != -1 else node.text.decode("utf-8")
        
        # Generic implementation for other languages
        return self._get_symbol_name(node, language) or node.type




    def _get_symbol_name(self, node: Node, language: Language) -> Optional[str]:
        """Extract the name of a symbol (function, class, etc.)"""
        try:
            # Try to find name field
            name_node = node.child_by_field_name("name")
            if name_node:
                return name_node.text.decode("utf-8")
                
            # Fallback: look for identifier in children
            for child in node.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8")
        except Exception:
            pass
        return None

    def _split_chunk_improved(self, code: str, file_path: str, language: Language, symbol_type: str, 
                        start_line: int, context: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """Improved chunk splitting with better line tracking and infinite loop prevention"""
        lines = code.splitlines()
        tokens = self.encoding.encode(code)
        
        if len(tokens) <= self.chunk_size:
            # If the code is small enough, just return it as a single chunk
            yield self._create_chunk_dict(
                file_path, language, symbol_type, code, 
                start_line, start_line + len(lines) - 1, len(tokens), context
            )
            return
        
        chunk_start = 0
        chunk_num = 0
        max_chunks = 100  # Safety limit to prevent infinite loops
        
        while chunk_start < len(tokens) and chunk_num < max_chunks:
            chunk_end = min(chunk_start + self.chunk_size, len(tokens))
            
            # Ensure we're making progress - if chunk_end <= chunk_start, break
            if chunk_end <= chunk_start:
                print(f"⚠️ Breaking split loop: chunk_end ({chunk_end}) <= chunk_start ({chunk_start}) for {file_path}")
                break
                
            chunk_tokens = tokens[chunk_start:chunk_end]
            
            # Safety check for empty chunks
            if not chunk_tokens:
                print(f"⚠️ Empty chunk tokens at position {chunk_start} for {file_path}")
                break
                
            try:
                chunk_code = self.encoding.decode(chunk_tokens)
            except Exception as e:
                print(f"⚠️ Error decoding chunk tokens for {file_path}: {e}")
                break
            
            # More accurate line number calculation
            prefix_code = self.encoding.decode(tokens[:chunk_start]) if chunk_start > 0 else ""
            prefix_lines = prefix_code.count('\n')
            chunk_lines = chunk_code.count('\n')
            
            chunk_start_line = start_line + prefix_lines
            chunk_end_line = chunk_start_line + chunk_lines
            
            # Add context about this being a split chunk
            split_context = context.copy()
            split_context["is_split"] = True
            split_context["split_index"] = chunk_num
            split_context["original_symbol"] = symbol_type
            
            yield self._create_chunk_dict(
                file_path, language, f"split_{symbol_type}", chunk_code, 
                chunk_start_line, chunk_end_line, len(chunk_tokens), split_context
            )
            
            # Calculate next chunk start with overlap, ensuring we make progress
            next_start = chunk_end - self.chunk_overlap
            
            # Ensure we're making progress - if next_start <= chunk_start, advance by at least 1 token
            if next_start <= chunk_start:
                next_start = chunk_start + max(1, self.chunk_size // 2)  # Advance by at least half chunk size
                print(f"⚠️ Forced advance from {chunk_start} to {next_start} for {file_path}")
            
            chunk_start = next_start
            chunk_num += 1
            
            # Additional safety check
            if chunk_num >= max_chunks:
                print(f"⚠️ Reached maximum chunk limit ({max_chunks}) for symbol in {file_path}")
                break
        
        if chunk_num >= max_chunks:
            print(f"⚠️ Split operation hit safety limit for {file_path} - may have truncated content")



    def _chunk_document(self, content: str, file_path: str, doc_type: str, last_modified : float) -> List[Dict[str, Any]]:
        """Enhanced document chunking for markdown, rst, etc."""
        chunks = []
        
        if doc_type == "markdown":
            sections = []
            current_section = []
            in_code_fence = False
            fence_char = None
            
            for i, line in enumerate(content.splitlines()):
                # Detect code fences
                if line.startswith("```") or line.startswith("~~~"):
                    if in_code_fence and line.startswith(fence_char):
                        in_code_fence = False
                        fence_char = None
                    else:
                        in_code_fence = True
                        fence_char = line[:3]

                # preserve code blocks as single chunks
                if in_code_fence:
                    current_section.append(line)
                    continue
                
                # Only split outside code fences
                if not in_code_fence and line.startswith("#"):
                    if current_section:
                        sections.append("\n".join(current_section))
                        current_section = []
                current_section.append(line)
            
            if current_section:
                sections.append("\n".join(current_section))
            
            # Create chunks for each section
            for i, section in enumerate(sections):
                token_count = len(self.encoding.encode(section))
                chunks.append(self._create_chunk_dict(
                    file_path, doc_type, "section", section, 
                    0, 0, token_count, {"section_index": i},
                    last_modified=last_modified
                ))
        else:
            # Simple paragraph-based chunking for other document types
            paragraphs = content.split('\n\n')
            current_chunk = ""
            current_tokens = 0
            
            for para in paragraphs:
                para_tokens = len(self.encoding.encode(para))
                if current_tokens + para_tokens > self.chunk_size and current_chunk:
                    chunks.append(self._create_chunk_dict(
                        file_path, doc_type, "paragraph", current_chunk.strip(), 
                        0, 0, current_tokens, {}
                    ))
                    current_chunk = para
                    current_tokens = para_tokens
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
                    current_tokens += para_tokens
            
            if current_chunk.strip():
                chunks.append(self._create_chunk_dict(
                    file_path, doc_type, "paragraph", current_chunk.strip(), 
                    0, 0, current_tokens, {}
                ))
        
        return chunks

    def _split_markdown_by_headers(self, content: str) -> List[Dict[str, Any]]:
        """Split markdown content by headers while preserving hierarchy"""
        lines = content.splitlines()
        sections = []
        current_section = {"content": "", "start_line": 0, "end_line": 0}
        
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                # Save previous section
                if current_section["content"].strip():
                    current_section["end_line"] = i - 1
                    sections.append(current_section)
                
                # Start new section
                level = len(line) - len(line.lstrip('#'))
                header = line.strip('#').strip()
                current_section = {
                    "content": line,
                    "header": header,
                    "level": level,
                    "start_line": i,
                    "end_line": i
                }
            else:
                current_section["content"] += "\n" + line
        
        # Add final section
        if current_section["content"].strip():
            current_section["end_line"] = len(lines) - 1
            sections.append(current_section)
        
        return sections

    def _split_large_text(self, text: str, file_path: str, language: str, symbol_type: str) -> Generator[Dict[str, Any], None, None]:
        """Split large text chunks with overlap"""
        tokens = self.encoding.encode(text)
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            yield self._create_chunk_dict(
                file_path, language, symbol_type, chunk_text, 
                0, 0, len(chunk_tokens), {"is_split": True}
            )

    def _chunk_config(self, content: str, file_path: str, config_type: str, last_modified : float = None) -> List[Dict[str, Any]]:
        """Handle configuration files"""
        token_count = len(self.encoding.encode(content))
        
        if token_count <= self.chunk_size:
            return [self._create_chunk_dict(
                file_path, config_type, "config_file", content, 
                0, len(content.splitlines()), token_count,
                {"file_type": config_type}, last_modified=last_modified
            )]
        else:
            return list(self._split_large_text(content, file_path, config_type, "config_section"))

    def _create_chunk_dict(self, file_path: str, language: Language, symbol_type: str, code: str, 
                      start_line: int, end_line: int, token_count: int, context: Dict[str, Any], 
                      is_overview: bool = False, last_modified : float = None, symbol_signature : str = None) -> Dict[str, Any]:
        """Create a simplified chunk dictionary - only essential data for embeddings"""
        
        # What goes into embedding (clean, focused content)
        embed_content = code.strip()
                
        return {
            "id": str(uuid.uuid4()),
            "content": embed_content,
            "file_path": file_path,
            "language": language.name if hasattr(language, 'name') else str(language),
            "symbol_type": symbol_type,
            "symbol_name": context.get("symbol_name"),
            "symbol_signature": symbol_signature,  # NEW: Full signature
            "start_line": start_line + 1,
            "end_line": end_line + 1,
            "token_count": token_count,
            "parent_class": context.get("parent_class"),
            "is_overview": is_overview,
            "module": context.get("package_path", ""),  # NEW: Module path
            "last_modified": datetime.datetime.fromtimestamp(last_modified).isoformat() if last_modified else None  # NEW: Timestamp
        }

def main():
    parser = argparse.ArgumentParser(description='Enhanced repository parser for embeddings')
    parser.add_argument('repo_path', help='Path to the repository to parse')
    parser.add_argument('--output', '-o', default='parsed_repo.parquet', help='Output file path (default: parsed_repo.parquet)')
    parser.add_argument('--chunk-size', type=int, default=512, help='Maximum tokens per chunk (default: 512)')
    parser.add_argument('--chunk-overlap', type=int, default=64, help='Token overlap between chunks (default: 64)')
    parser.add_argument('--ignore-dirs', nargs='*', default=[], help='Additional directories to ignore')
    parser.add_argument('--ignore-files', nargs='*', default=[], help='Additional files to ignore')
    parser.add_argument('--summary-only', action='store_true', help='Only show repository summary without parsing')
    parser.add_argument('--format', choices=['parquet', 'json', 'csv'], default='parquet', help='Output format (default: parquet)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate repository path
    if not os.path.exists(args.repo_path):
        print(f"❌ Error: Repository path '{args.repo_path}' does not exist")
        return 1
    
    if not os.path.isdir(args.repo_path):
        print(f"❌ Error: '{args.repo_path}' is not a directory")
        return 1
    
    # Initialize parser
    repo_parser = RepoParser(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        ignore_dirs=args.ignore_dirs,
        ignore_files=args.ignore_files
    )
    
    try:
        # Display repository summary
        print(f"🔍 Analyzing repository: {os.path.abspath(args.repo_path)}")
        repo_parser.display_summary(args.repo_path)
        
        if args.summary_only:
            print("📋 Summary complete. Use without --summary-only to parse the repository.")
            return 0
        
        # Parse the repository
        print(f"\n🚀 Starting repository parsing...")
        chunks = repo_parser.parse_repo(args.repo_path)
        
        if not chunks:
            print("⚠️  No chunks generated. Check if the repository contains supported file types.")
            return 1
        
        # Prepare output data
        df = pd.DataFrame(chunks)
        
        # Save to specified format
        output_path = args.output
        if args.format == 'parquet':
            if not output_path.endswith('.parquet'):
                output_path = os.path.splitext(output_path)[0] + '.parquet'
            
            # Convert to PyArrow table for better performance
            table = pa.Table.from_pandas(df)
            pq.write_table(table, output_path, compression='snappy')
            
        elif args.format == 'json':
            if not output_path.endswith('.json'):
                output_path = os.path.splitext(output_path)[0] + '.json'
            df.to_json(output_path, orient='records', indent=2)
            
        elif args.format == 'csv':
            if not output_path.endswith('.csv'):
                output_path = os.path.splitext(output_path)[0] + '.csv'
            df.to_csv(output_path, index=False)
        
        # Display final statistics
        print(f"\n✅ Parsing complete!")
        print(f"📊 Generated {len(chunks):,} chunks")
        print(f"💾 Saved to: {os.path.abspath(output_path)}")
        print(f"📁 Output format: {args.format.upper()}")
        
        if args.verbose:
            print(f"\n📈 Detailed Statistics:")
            print(f"  Total tokens: {df['token_count'].sum():,}")
            print(f"  Average tokens per chunk: {df['token_count'].mean():.1f}")
            print(f"  Median tokens per chunk: {df['token_count'].median():.1f}")
            print(f"  Largest chunk: {df['token_count'].max():,} tokens")
            print(f"  Smallest chunk: {df['token_count'].min():,} tokens")
            
            print(f"\n🔤 Language breakdown:")
            lang_stats = df['language'].value_counts()
            for lang, count in lang_stats.items():
                percentage = (count / len(df)) * 100
                print(f"  {lang.upper():>12}: {count:>5} chunks ({percentage:>5.1f}%)")
            
            print(f"\n🔍 Symbol type breakdown:")
            symbol_stats = df['symbol_type'].value_counts()
            for symbol, count in symbol_stats.head(10).items():
                percentage = (count / len(df)) * 100
                print(f"  {symbol:>15}: {count:>5} chunks ({percentage:>5.1f}%)")
        
        print(f"\n🎯 Ready for embedding with Voyage Code 3!")
        print(f"💡 Tip: Use the 'code' column for embedding and store other metadata for context")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Parsing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error during parsing: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())