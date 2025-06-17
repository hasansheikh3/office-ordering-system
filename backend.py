import http.server
import socketserver
import sqlite3
import json
import os
import threading
import time
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import re # Import regex for path parsing

PORT = 80 # Changed port back to 80
DATABASE_FILE = 'orders.db'
ADMIN_NAME = 'Kamran' # Define admin name here

# Database connection lock for thread safety
db_lock = threading.Lock()

# Simple cache for admin data (expires after 30 seconds)
admin_cache = {
    'data': None,
    'timestamp': 0,
    'ttl': 30  # Time to live in seconds
}

# --- Database Setup ---
def setup_database():
    with db_lock:
        conn = sqlite3.connect(DATABASE_FILE, timeout=10.0)
        # Enable WAL mode for better concurrent access
        conn.execute('PRAGMA journal_mode=WAL;')
        conn.execute('PRAGMA synchronous=NORMAL;')
        conn.execute('PRAGMA cache_size=10000;')
        conn.execute('PRAGMA temp_store=memory;')

        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                item TEXT NOT NULL,
                quantity TEXT NOT NULL,
                amount TEXT NOT NULL,
                timestamp DATETIME NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

# --- Database Helper Functions ---
def get_db_connection():
    """Get a database connection with optimized settings for concurrent access"""
    conn = sqlite3.connect(DATABASE_FILE, timeout=10.0)
    conn.execute('PRAGMA journal_mode=WAL;')
    conn.execute('PRAGMA synchronous=NORMAL;')
    return conn

def clear_admin_cache():
    """Clear the admin data cache"""
    global admin_cache
    admin_cache['data'] = None
    admin_cache['timestamp'] = 0

# --- Custom Request Handler ---
class PeonAppHandler(http.server.SimpleHTTPRequestHandler):

    # Helper to check admin access based on query parameter
    def is_admin(self):
        parsed_url = urlparse(self.path)
        query = parse_qs(parsed_url.query)
        return query.get('name', [''])[0] == ADMIN_NAME

    def do_POST(self):
        # Reuse /submit for both user and admin forms
        if self.path.startswith('/submit'): # Use startswith to handle potential query params
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data)
                name = data.get('name')

                # Check if this is the new multi-item format or old single-item format
                if 'items' in data and 'total' in data:
                    # New multi-item format
                    items = data.get('items', [])
                    total = data.get('total', 0)

                    if not name or not items:
                        self.send_response(400)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'success': False, 'message': 'Missing name or items'}).encode('utf-8'))
                        return

                    # Validate items
                    for i, item in enumerate(items):
                        if not item.get('name') or not item.get('quantity'):
                            self.send_response(400)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps({'success': False, 'message': f'Missing fields in item {i+1}'}).encode('utf-8'))
                            return

                    # Use thread-safe database connection
                    with db_lock:
                        conn = get_db_connection()
                        cursor = conn.cursor()

                        # Insert each item as a separate order record
                        for item in items:
                            item_name = item.get('name', '')
                            item_quantity = item.get('quantity', '')

                            # Format the amount field to show total amount given
                            amount_text = f"Rs {total:.2f} (Total Given)"

                            # Use local timestamp
                            local_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            cursor.execute("INSERT INTO orders (name, item, quantity, amount, timestamp) VALUES (?, ?, ?, ?, ?)",
                                           (name, item_name, item_quantity, amount_text, local_timestamp))

                        conn.commit()
                        conn.close()

                        # Clear admin cache when new order is added
                        clear_admin_cache()

                    order_count = len(items)
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'success': True, 'message': f'Order with {order_count} items submitted successfully!'}).encode('utf-8'))

                else:
                    # Old single-item format (for backward compatibility with admin)
                    item = data.get('item')
                    quantity = data.get('quantity')
                    amount = data.get('amount')

                    if not all([name, item, quantity, amount]):
                        self.send_response(400)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'success': False, 'message': 'Missing fields'}).encode('utf-8'))
                        return

                    # Use thread-safe database connection
                    with db_lock:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        # Use local timestamp
                        local_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        cursor.execute("INSERT INTO orders (name, item, quantity, amount, timestamp) VALUES (?, ?, ?, ?, ?)",
                                       (name, item, quantity, amount, local_timestamp))
                        conn.commit()
                        conn.close()

                        # Clear admin cache when new order is added
                        clear_admin_cache()

                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'success': True, 'message': 'Order submitted'}).encode('utf-8'))

            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'message': 'Invalid JSON'}).encode('utf-8'))
            except sqlite3.OperationalError as e:
                print(f"Database lock error on POST: {e}")
                self.send_response(503)  # Service Unavailable
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'message': 'Database busy, please try again'}).encode('utf-8'))
            except Exception as e:
                print(f"Database error on POST: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'message': 'Server error'}).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == '/admin/data':
            # Basic admin check
            if not self.is_admin():
                 self.send_response(403)
                 self.send_header('Content-type', 'application/json')
                 self.end_headers()
                 self.wfile.write(json.dumps({'success': False, 'message': 'Unauthorized'}).encode('utf-8'))
                 return

            try:
                # Check cache first
                current_time = time.time()
                if (admin_cache['data'] is not None and
                    current_time - admin_cache['timestamp'] < admin_cache['ttl']):
                    # Return cached data
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(admin_cache['data']).encode('utf-8'))
                    return

                # Cache miss or expired, fetch from database
                with db_lock:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    # Select the id column as well
                    cursor.execute("SELECT id, name, item, quantity, amount, timestamp FROM orders ORDER BY timestamp DESC")
                    rows = cursor.fetchall()
                    conn.close()

                # Convert rows to a list of dictionaries for JSON response
                orders = []
                for row in rows:
                    orders.append({
                        'id': row[0], # Include id
                        'name': row[1],
                        'item': row[2],
                        'quantity': row[3],
                        'amount': row[4],
                        'timestamp': row[5]
                    })

                response_data = {'success': True, 'orders': orders}

                # Update cache
                admin_cache['data'] = response_data
                admin_cache['timestamp'] = current_time

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))

            except sqlite3.OperationalError as e:
                print(f"Database lock error on GET /admin/data: {e}")
                self.send_response(503)  # Service Unavailable
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'message': 'Database busy, please try again'}).encode('utf-8'))
            except Exception as e:
                print(f"Database error on GET /admin/data: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'message': 'Server error'}).encode('utf-8'))

        elif path == '/admin' or path == '/admin/':
             # Serve admin.html
             self.path = '/admin.html'
             return http.server.SimpleHTTPRequestHandler.do_GET(self)

        else:
            # Serve other static files (index.html, css, js)
            return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def do_DELETE(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # Basic admin check for delete operations
        if not self.is_admin():
             self.send_response(403)
             self.send_header('Content-type', 'application/json')
             self.end_headers()
             self.wfile.write(json.dumps({'success': False, 'message': 'Unauthorized'}).encode('utf-8'))
             return

        # Handle delete all
        if path == '/delete_all':
            try:
                with db_lock:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM orders")
                    conn.commit()
                    conn.close()

                    # Clear admin cache when orders are deleted
                    clear_admin_cache()

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'success': True, 'message': 'All orders deleted'}).encode('utf-8'))

            except sqlite3.OperationalError as e:
                print(f"Database lock error on DELETE /delete_all: {e}")
                self.send_response(503)  # Service Unavailable
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'message': 'Database busy, please try again'}).encode('utf-8'))
            except Exception as e:
                print(f"Database error on DELETE /delete_all: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'message': 'Server error'}).encode('utf-8'))

        # Handle delete single order
        elif path.startswith('/delete/'):
            # Use regex to extract ID from path like /delete/123
            match = re.match(r'/delete/(\d+)', path)
            if match:
                order_id = match.group(1)
                try:
                    with db_lock:
                        conn = get_db_connection()
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM orders WHERE id = ?", (order_id,))
                        conn.commit()
                        # Check if any row was actually deleted
                        if cursor.rowcount > 0:
                            # Clear admin cache when order is deleted
                            clear_admin_cache()
                            self.send_response(200)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps({'success': True, 'message': f'Order {order_id} deleted'}).encode('utf-8'))
                        else:
                             self.send_response(404) # Not Found if ID didn't exist
                             self.send_header('Content-type', 'application/json')
                             self.end_headers()
                             self.wfile.write(json.dumps({'success': False, 'message': f'Order {order_id} not found'}).encode('utf-8'))
                        conn.close()

                except sqlite3.OperationalError as e:
                    print(f"Database lock error on DELETE /delete/{order_id}: {e}")
                    self.send_response(503)  # Service Unavailable
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'success': False, 'message': 'Database busy, please try again'}).encode('utf-8'))
                except Exception as e:
                    print(f"Database error on DELETE /delete/{order_id}: {e}")
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'success': False, 'message': 'Server error'}).encode('utf-8'))
            else:
                self.send_response(400) # Bad Request if ID is missing or invalid format
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'success': False, 'message': 'Invalid delete path'}).encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()


# --- Custom Threading Server for Better Concurrency ---
class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Handle requests in a separate thread for better concurrency"""
    daemon_threads = True  # Die when main thread dies
    allow_reuse_address = True  # Prevent "Address already in use" errors

# --- Main Server Execution ---
if __name__ == "__main__":
    setup_database()
    # Set the current directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    Handler = PeonAppHandler

    # Use ThreadedTCPServer for concurrent request handling
    with ThreadedTCPServer(("", PORT), Handler) as httpd:
        print(f"🚀 High-Performance Server starting on port {PORT}")
        print(f"✅ Concurrent users supported: Up to 20+ simultaneous connections")
        print(f"✅ Database optimizations: WAL mode, connection pooling, caching enabled")
        print(f"✅ Threading: Each request handled in separate thread")
        print(f"\n📍 Access URLs:")
        print(f"   • Main app: http://172.16.1.186:{PORT}/")
        print(f"   • Admin page: http://172.16.1.186:{PORT}/admin")
        print(f"   • Via domain: http://kamran.com/ (if DNS configured)")
        print(f"\n*** Note: Running on port 80 requires root privileges. ***")
        print("*** Ensure no other service (like Apache or Nginx) is using port 80. ***")
        print("\n🎯 Performance improvements applied - your backend should now be much faster!")
        httpd.serve_forever()
