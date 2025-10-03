from main import LinkedListDB, INPUT_CSV, OUTPUT_CSV_BUBBLE

def run_bubble():
    db = LinkedListDB.load_from_csv_recursive(INPUT_CSV)
    db.sync_with_csv(INPUT_CSV)
    steps, dt = db.bubble_sort(db.key_name, reverse=False)
    print(f"Bubble sort: steps={steps}, seconds={dt:.6f}")
    db.export_to_csv_recursive(OUTPUT_CSV_BUBBLE)

if __name__ == "__main__":
    run_bubble()
