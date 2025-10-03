from main import LinkedListDB, INPUT_CSV

OUTPUT_CSV_MERGE = "output_merge.csv"

def run_merge():
    db = LinkedListDB.load_from_csv_recursive(INPUT_CSV)
    db.sync_with_csv(INPUT_CSV)
    steps, dt = db.merge_sort(db.key_name, reverse=False)
    print(f"Merge sort: steps={steps}, seconds={dt:.6f}")
    db.export_to_csv_recursive(OUTPUT_CSV_MERGE)

if __name__ == "__main__":
    run_merge()
