import csv
import os
import time
from typing import Dict, List, Optional
import subprocess
import sys

# Performance measurement structure
class PerformanceMetrics:
    def __init__(self, algorithm: str, execution_time: float, cpu_usage: float, 
                 memory_usage: float, disk_space_usage: int, timestamp: str):
        self.algorithm = algorithm
        self.execution_time = execution_time
        self.cpu_usage = cpu_usage
        self.memory_usage = memory_usage
        self.disk_space_usage = disk_space_usage
        self.timestamp = timestamp

# Node definition for singly linked list
class Node:
    """A node in a singly linked list, storing a dictionary of key-value pairs and a reference to the next node."""
    def __init__(self, data: Dict[str, str]):
        self.data = data
        self.next = None

# Implementation of singly linked list
class SinglyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    # Append a node to the last of the list
    def add_node(self, data: Dict[str, str]) -> Node:
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1
        return new_node

    # Remove a node at the specific index
    def remove_node(self, index: int) -> bool:
        if index < 0 or index >= self.size:
            return False
        if index == 0:
            self.head = self.head.next
            if self.head is None:
                self.tail = None
            self.size -= 1
            return True
        
        current = self.head
        for _ in range(index - 1):
            current = current.next
        
        current.next = current.next.next
        if current.next is None:
            self.tail = current
        self.size -= 1
        return True

    # Return the node at the given index (0-based). Returns None if the index is invalid
    def get_node_at(self, index: int) -> Optional[Node]:
        if index < 0 or index >= self.size:
            return None
        current = self.head
        for _ in range(index):
            current = current.next
        return current

    # Print all the nodes in the linked list
    def display_list(self):
        current = self.head
        while current is not None:
            print(current.data)
            current = current.next

# Memory database built on a singly linked list backed by CSV file
class MemoryDatabase:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.data_list = SinglyLinkedList()
        self.last_modified_time = 0.0
        self.headers = []
        self.performance_logs = []

# Recursive function to load CSV data into linked list
def load_csv(db: MemoryDatabase, rows: List[List[str]], index: int):
    if index >= len(rows):
        return

    row = rows[index]
    # Create dictionary with headers as keys
    row_data = {}
    for i, h in enumerate(db.headers):
        if i < len(row):
            row_data[h] = row[i]
        else:
            row_data[h] = ""
    db.data_list.add_node(row_data)
    load_csv(db, rows, index + 1)

# Load data from CSV file
def load_data(db: MemoryDatabase):
    if not os.path.isfile(db.csv_file):
        print(f"CSV file not found: {db.csv_file}")
        return
    
    try:
        with open(db.csv_file, 'r', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    if not rows:
        print("CSV file is empty")
        return

    # First row = headers
    db.headers = [str(x) for x in rows[0]]
    data_rows = rows[1:]

    db.data_list = SinglyLinkedList()
    if data_rows:
        load_csv(db, data_rows, 0)

    db.last_modified_time = os.stat(db.csv_file).st_mtime
    print(f"Successfully loaded {db.data_list.size} records from {db.csv_file}")

# Recursively write linked list nodes to an open CSV file handle
def export_csv(file, node: Optional[Node], headers: List[str]):
    if node is None:
        return
    row = [node.data.get(h, "") for h in headers]
    file.write(','.join(row) + '\n')
    export_csv(file, node.next, headers)

# Export the linked list to a CSV file with headers and data
def export_to_csv(db: MemoryDatabase, output_file: str = "python_result.csv"):
    if not db.headers:
        print("No data to export")
        return
    
    try:
        with open(output_file, 'w', newline='') as file:
            file.write(','.join(db.headers) + '\n')
            export_csv(file, db.data_list.head, db.headers)
        print(f"Exported {db.data_list.size} records to {output_file}")
    except Exception as e:
        print(f"Error exporting to CSV: {e}")

# Print the database contents with headers and row values
def display_data(db: MemoryDatabase):
    print("\nCurrent Memory Database Contents:")
    print(f"Total records: {db.data_list.size}")
    if db.headers:
        print(" | ".join(db.headers))
    
    current = db.data_list.head
    while current is not None:
        row_values = [current.data.get(h, "") for h in db.headers]
        print(" | ".join(row_values))
        current = current.next

# Check for the modification of underlying CSV
def check_changes(db: MemoryDatabase) -> bool:
    if not os.path.isfile(db.csv_file):
        return False
    
    current_modified = os.stat(db.csv_file).st_mtime
    if current_modified > db.last_modified_time:
        print("CSV file changed. Reloading data.")
        load_data(db)
        export_to_csv(db, "python_result.csv")
        return True
    return False

# Continuously monitor the CSV file for changes at interval seconds
def monitor_changes(db: MemoryDatabase, interval: int):
    print(f"Monitoring {db.csv_file} for changes. (Ctrl+C to stop)")
    try:
        while True:
            check_changes(db)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

# Prompt the user to enter values for each column header and append a new row to the linked list
def add_row(db: MemoryDatabase):
    if not db.headers:
        print("No headers available. Load data first.")
        return
    
    row_data = {}
    for h in db.headers:
        value = input(f"Enter {h}: ")
        row_data[h] = value
    
    db.data_list.add_node(row_data)
    export_to_csv(db, "python_result.csv")
    print("Row added to memory database")

# Prompt the user for a row index and remove that row from the database
def remove_row(db: MemoryDatabase):
    if db.data_list.size == 0:
        print("No records to remove.")
        return
    
    try:
        index = int(input(f"Enter row number to remove (0-{db.data_list.size-1}): "))
        if index < 0 or index >= db.data_list.size:
            print("Invalid row number.")
            return
        
        row_data = db.data_list.get_node_at(index)
        row_values = [row_data.data.get(header, "") for header in db.headers]
        print(f"Record to be removed: {' | '.join(row_values)}")
        
        confirm = input("Are you sure you want to remove this record? (y/n): ").lower()
        if confirm == 'y':
            if db.data_list.remove_node(index):
                export_to_csv(db, "python_result.csv")
                print("Record removed successfully.")
    except ValueError:
        print("Invalid input. Please enter a number.")

# Validate whether the specified column exist in headers or not
def validate_key(db: MemoryDatabase, key: str) -> bool:
    if not db.headers or key not in db.headers:
        print(f"Error: Column '{key}' is not found in CSV headers.")
        if db.headers:
            print(f"Available Columns: {', '.join(db.headers)}")
        return False
    return True

# Helper function for comparison
def compare_values(a: str, b: str, order: str = "ASC") -> bool:
    # Try to convert to numeric for proper comparison
    try:
        val1 = 0.0 if not a else float(a)
        val2 = 0.0 if not b else float(b)
        if order == "ASC":
            return val1 <= val2
        else:
            return val1 >= val2
    except ValueError:
        # If conversion fails to numeric, compare as strings
        if order == "ASC":
            return a <= b
        else:
            return a >= b

# Bubble sort for the linked list
def bubble_sort(db: MemoryDatabase, key: str, order: str = "ASC") -> bool:
    if not validate_key(db, key):
        return False
    
    if db.data_list.size <= 1:
        print("Not enough data to sort.")
        return True
    
    # Convert linked list to array for easier sorting
    nodes = []
    current = db.data_list.head
    while current is not None:
        nodes.append(current)
        current = current.next
    
    # Bubble sort algorithm
    n = len(nodes)
    for i in range(n - 1):
        swapped = False
        for j in range(n - i - 1):
            value1 = nodes[j].data.get(key, "")
            value2 = nodes[j + 1].data.get(key, "")
            if not compare_values(value1, value2, order):
                nodes[j].data, nodes[j + 1].data = nodes[j + 1].data, nodes[j].data
                swapped = True
        if not swapped:
            break
    
    print(f"Bubble sort completed on column: {key} ({order})")
    return True

# Insertion sort for the linked list
def insertion_sort(db: MemoryDatabase, key: str, order: str = "ASC") -> bool:
    if not validate_key(db, key):
        return False
    
    if db.data_list.size <= 1:
        print("Not enough data to sort.")
        return True
    
    # Convert to array for sorting as we are using a singly linked list
    nodes = []
    current = db.data_list.head
    while current is not None:
        nodes.append(current)
        current = current.next
    
    # Insertion sort algorithm
    for i in range(1, len(nodes)):
        j = i
        while j > 0:
            value1 = nodes[j - 1].data.get(key, "")
            value2 = nodes[j].data.get(key, "")
            if not compare_values(value1, value2, order):
                nodes[j - 1].data, nodes[j].data = nodes[j].data, nodes[j - 1].data
                j -= 1
            else:
                break
    
    print(f"Insertion sort completed on column: {key} ({order})")
    return True

# Merge sort implementation for linked list
def merge_sort(db: MemoryDatabase, key: str, order: str = "ASC") -> bool:
    if not validate_key(db, key):
        return False
    
    if db.data_list.size <= 1:
        print("Not enough data to sort.")
        return True
    
    # Convert linked list to array for easier sorting
    nodes = []
    current = db.data_list.head
    while current is not None:
        nodes.append(current)
        current = current.next
    
    # Recursive merge sort function
    def merge_sort_recursive(arr):
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = merge_sort_recursive(arr[:mid])
        right = merge_sort_recursive(arr[mid:])
        
        return merge(left, right)
    
    # Merge function
    def merge(left, right):
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            value1 = left[i].data.get(key, "")
            value2 = right[j].data.get(key, "")
            
            if compare_values(value1, value2, order):
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        # Append remaining elements
        result.extend(left[i:])
        result.extend(right[j:])
        
        return result
    
    # Perform merge sort
    sorted_nodes = merge_sort_recursive(nodes)
    
    # Update the linked list with sorted nodes
    db.data_list = SinglyLinkedList()
    for node in sorted_nodes:
        db.data_list.add_node(node.data)
    
    print(f"Merge sort is implemented on column: {key}")
    return True

# Quick sort implementation for linked list
def quick_sort(db: MemoryDatabase, key: str, order: str = "ASC") -> bool:
    if not validate_key(db, key):
        return False
    
    if db.data_list.size <= 1:
        print("Not enough data to sort.")
        return True
    
    # Convert linked list to array for easier sorting
    nodes = []
    current = db.data_list.head
    while current is not None:
        nodes.append(current)
        current = current.next
    
    # Recursive quick sort function
    def quick_sort_recursive(arr):
        if len(arr) <= 1:
            return arr
        
        pivot_index = len(arr) // 2
        pivot = arr[pivot_index]
        pivot_value = pivot.data.get(key, "")
        
        left = []
        right = []
        
        for i, node in enumerate(arr):
            if i != pivot_index:
                node_value = node.data.get(key, "")
                if compare_values(node_value, pivot_value, order):
                    left.append(node)
                else:
                    right.append(node)
        
        return quick_sort_recursive(left) + [pivot] + quick_sort_recursive(right)
    
    # Perform quick sort
    sorted_nodes = quick_sort_recursive(nodes)
    
    # Update the linked list with sorted nodes
    db.data_list = SinglyLinkedList()
    for node in sorted_nodes:
        db.data_list.add_node(node.data)
    
    print(f"Quick sort is implemented on column: {key}")
    return True

# Choose a column from header for implementing the sorting algorithm
def choose_sort_key(db: MemoryDatabase) -> Optional[str]:
    if not db.headers:
        print("No data is available. Please load the data first.")
        return None
    
    print(f"\nAvailable columns: {', '.join(db.headers)}")
    key_input = input("Enter a column name to sort: ")
    key = key_input.strip()
    
    if key not in db.headers:
        print(f"Error: '{key}' is not a valid column name.")
        return None
    
    return key

# Choose sorting order
def choose_sort_order() -> str:
    print("\nChoose sorting order:")
    print("1. Ascending (ASC)")
    print("2. Descending (DSC)")
    order_choice = input("Enter your choice (1 or 2): ").strip()
    
    if order_choice in ["1", "asc", "ASC"]:
        return "ASC"
    elif order_choice in ["2", "dsc", "DSC"]:
        return "DSC"
    else:
        print("Invalid choice. Using Ascending as default.")
        return "ASC"

# Options of sorting algorithm to implement
def opt_sorting_algorithm() -> str:
    print("\nChoose sorting algorithm for implementation in memory database:")
    print("1. Bubble Sort")
    print("2. Insertion Sort")
    print("3. Merge Sort")
    print("4. Quick Sort")
    print("5. Compare All Algorithms (Computational Performance)")
    print("6. Execute SQL Query")
    
    algorithm_choice = input("Enter your choice (1, 2, 3, 4, 5 or 6): ").strip()
    
    if algorithm_choice == "1":
        return "bubble"
    elif algorithm_choice == "2":
        return "insertion"
    elif algorithm_choice == "3":
        return "merge"
    elif algorithm_choice == "4":
        return "quick"
    elif algorithm_choice == "5":
        return "compare"
    elif algorithm_choice == "6":
        return "sql"
    else:
        print("Invalid choice. Using Bubble Sort as default.")
        return "bubble"

# Export the sorted data to CSV
def export_sorted_csv(db: MemoryDatabase, output_file: str, sort_type: str, 
                     key: Optional[str] = None, order: str = "ASC") -> bool:
    if key is None:
        key = choose_sort_key(db)
        if key is None:
            return False
    
    if not validate_key(db, key):
        return False
    
    if sort_type == "bubble":
        bubble_sort(db, key, order)
    elif sort_type == "insertion":
        insertion_sort(db, key, order)
    elif sort_type == "merge":
        merge_sort(db, key, order)
    elif sort_type == "quick":
        quick_sort(db, key, order)
    else:
        print("Sort type is invalid. Please use bubble/insertion/merge/quick sort.")
        return False
    
    export_to_csv(db, output_file)
    print(f"Sorted data exported to {output_file} using {sort_type} sort on column: {key}")
    return True

# Dispatcher to call the correct sorting algorithm
def sort_database(db: MemoryDatabase, key: str, order: str, method: str) -> bool:
    if method == "bubble_sort":
        return bubble_sort(db, key, order)
    elif method == "insertion_sort":
        return insertion_sort(db, key, order)
    elif method == "merge_sort":
        return merge_sort(db, key, order)
    elif method == "quick_sort":
        return quick_sort(db, key, order)
    else:
        print(f"Unknown sorting method: {method}")
        return False

# Compare performance of different sorting algorithms
def compare_sorting_performance(db: MemoryDatabase, key: str):
    methods = ["bubble_sort", "insertion_sort", "merge_sort", "quick_sort"]
    
    for m in methods:
        print(f"\n--- {m} ---")
        
        # Create a copy of the database
        db_copy = MemoryDatabase(db.csv_file)
        load_data(db_copy)
        
        t_start = time.time()
        sort_database(db_copy, key, "ASC", m)
        t_end = time.time()
        
        exec_time = t_end - t_start
        print(f"Execution time for {m}: {exec_time:.6f} seconds")

        # Capture system metrics (platform-dependent)
        try:
            if sys.platform == "linux":
                subprocess.run(["vmstat", "1", "2"], capture_output=True)
                subprocess.run(["iostat", "-dx", "1", "2"], capture_output=True)
                subprocess.run(["pidstat", "1", "2"], capture_output=True)
                subprocess.run(["df", "-h"], capture_output=True)
        except Exception as e:
            print(f"System metrics collection failed: {e}")

# SQL Query Parser Function
def parse_sql_query(db: MemoryDatabase, query: str) -> bool:
    try:
        # Convert to uppercase for case-insensitive parsing of keywords only
        query_upper = query.strip().upper()
        
        # Remove extra spaces and parse
        query_upper = ' '.join(query_upper.split())
        
        # Parse the SQL-like query
        # Expected format: SELECT columns FROM t1 ORDER BY column ASC/DSC WITH algorithm
        
        # Extract select clause
        if not query_upper.startswith("SELECT "):
            print("Error: Query must start with SELECT")
            return False
        
        # Extract from clause
        if " FROM " not in query_upper:
            print("Error: Missing FROM clause")
            return False
        from_index = query_upper.index(" FROM ")
        
        # Extract order by clause
        if " ORDER BY " not in query_upper:
            print("Error: Missing ORDER BY clause")
            return False
        order_by_index = query_upper.index(" ORDER BY ")
        
        # Extract with clause
        if " WITH " not in query_upper:
            print("Error: Missing WITH clause")
            return False
        with_index = query_upper.index(" WITH ")
        
        # Extract components - use original case for column names
        columns_part = query[7:from_index]  # After "SELECT " (original case)
        from_part = query_upper[from_index + 6:order_by_index]  # After "FROM "
        order_by_part = query[order_by_index + 10:with_index]  # After "ORDER BY " (original case)
        with_part = query_upper[with_index + 6:]  # After "WITH "
        
        # Parse columns - preserve original case
        columns_str = columns_part.strip()
        if columns_str.upper() == "*":
            selected_columns = db.headers
        else:
            selected_columns = [col.strip() for col in columns_str.split(",")]
        
        # Validate table name
        table_name = from_part.strip()
        if table_name != "T1":
            print(f"Error: Unknown table '{table_name}'. Only 'T1' is supported.")
            return False
        
        # Parse ORDER BY clause - preserve original case for column name
        order_parts = order_by_part.strip().split()
        if len(order_parts) < 2:
            print("Error: Invalid ORDER BY clause. Expected: column ASC/DSC")
            return False
        
        sort_column = order_parts[0].strip()  # Keep original case
        sort_order = order_parts[1].strip().upper()  # Convert to uppercase for order
        
        if sort_order not in ["ASC", "DSC"]:
            print(f"Error: Invalid sort order '{sort_order}'. Use ASC or DSC.")
            return False
        
        # Parse algorithm
        algorithm = with_part.strip()
        valid_algorithms = ["BUBBLE_SORT", "INSERTION_SORT", "MERGE_SORT", "QUICK_SORT"]
        if algorithm not in valid_algorithms:
            print(f"Error: Invalid algorithm '{algorithm}'. Use: {', '.join(valid_algorithms)}")
            return False
        
        # Validate columns - use case-insensitive comparison
        for col in selected_columns:
            col_normalized = col.strip().lower()
            if not any(h.lower() == col_normalized for h in db.headers):
                print(f"Error: Column '{col}' is not found in CSV headers.")
                print(f"Available Columns: {', '.join(db.headers)}")
                return False
        
        # Validate sort column with case-insensitive comparison
        sort_column_normalized = sort_column.lower()
        matching_headers = [h for h in db.headers if h.lower() == sort_column_normalized]
        if not matching_headers:
            print(f"Error: Column '{sort_column}' is not found in CSV headers.")
            print(f"Available Columns: {', '.join(db.headers)}")
            return False
        
        # Find the actual column name from headers (case-sensitive)
        actual_sort_column = matching_headers[0]
        
        # Map algorithm name to function type
        algo_map = {
            "BUBBLE_SORT": "bubble",
            "INSERTION_SORT": "insertion", 
            "MERGE_SORT": "merge",
            "QUICK_SORT": "quick"
        }
        
        algo_type = algo_map[algorithm]
        
        # Execute the sort
        print("\nExecuting SQL Query:")
        print(f"Columns: {', '.join(selected_columns)}")
        print(f"Sort by: {actual_sort_column} {sort_order}")
        print(f"Algorithm: {algorithm}")
        
        # Create a copy for sorting to preserve original data
        db_copy = MemoryDatabase(db.csv_file)
        load_data(db_copy)
        
        # Perform sorting using actual column name
        if algo_type == "bubble":
            bubble_sort(db_copy, actual_sort_column, sort_order)
        elif algo_type == "insertion":
            insertion_sort(db_copy, actual_sort_column, sort_order)
        elif algo_type == "merge":
            merge_sort(db_copy, actual_sort_column, sort_order)
        elif algo_type == "quick":
            quick_sort(db_copy, actual_sort_column, sort_order)

        # Export result
        output_file = "python_sql_result.csv"
        export_to_csv(db_copy, output_file)
        
        # Display selected columns from sorted data
        print("\nQuery Results (first 10 rows):")
        # Find actual column names for display
        display_columns = []
        for col in selected_columns:
            col_normalized = col.strip().lower()
            matching_cols = [h for h in db.headers if h.lower() == col_normalized]
            actual_col = matching_cols[0] if matching_cols else col
            display_columns.append(actual_col)
        
        print(" | ".join(display_columns))
        
        current = db_copy.data_list.head
        count = 0
        while current is not None and count < 10:
            row_values = [current.data.get(col, "") for col in display_columns]
            print(" | ".join(row_values))
            current = current.next
            count += 1
        
        if count == 10:
            print("(Showing first 10 rows only)")
        
        print(f"\nFull results exported to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error parsing SQL query: {e}")
        print("Expected format: SELECT columns FROM t1 ORDER BY column ASC/DSC WITH algorithm")
        return False

# Interactive SQL query interface
def sql_query_interface(db: MemoryDatabase):
    print("SQL Query Interface")
    print("Supported syntax:")
    print("SELECT column1, column2 FROM t1 ORDER BY column_name ASC WITH BUBBLE_SORT")
    print("SELECT * FROM t1 ORDER BY age DSC WITH QUICK_SORT")
    print("Available algorithms: BUBBLE_SORT, INSERTION_SORT, MERGE_SORT, QUICK_SORT")
    print("Type 'exit' to return to main menu")
    
    while True:
        query = input("\nSQL> ").strip()
        if query.lower() == "exit":
            break
        elif not query:
            continue
        parse_sql_query(db, query)

# Main Execution
def main():
    # Initialize the memory database
    db = MemoryDatabase("student-data.csv")

    # Load initial data
    load_data(db)

    # Display loaded data
    display_data(db)

    # Export to new CSV
    export_to_csv(db, "python_result.csv")

    # Main interactive loop
    while True:
        # Ask user to choose a sorting algorithm
        sort_type = opt_sorting_algorithm()
        
        if sort_type == "bubble":
            key = choose_sort_key(db)
            if key is not None:
                order = choose_sort_order()
                export_sorted_csv(db, "python_bubble_sort.csv", "bubble", key, order)
        elif sort_type == "insertion":
            key = choose_sort_key(db)
            if key is not None:
                order = choose_sort_order()
                export_sorted_csv(db, "python_insertion_sort.csv", "insertion", key, order)
        elif sort_type == "merge":
            key = choose_sort_key(db)
            if key is not None:
                order = choose_sort_order()
                export_sorted_csv(db, "python_merge_sort.csv", "merge", key, order)
        elif sort_type == "quick":
            key = choose_sort_key(db)
            if key is not None:
                order = choose_sort_order()
                export_sorted_csv(db, "python_quick_sort.csv", "quick", key, order)
        elif sort_type == "compare":
            key = choose_sort_key(db)
            if key is not None:
                compare_sorting_performance(db, key)
        elif sort_type == "sql":
            sql_query_interface(db)
        else:
            print("Invalid selection.")
        
        # Ask if user wants to continue
        continue_choice = input("\nDo you want to perform another operation? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            break

if __name__ == "__main__":
    main()
