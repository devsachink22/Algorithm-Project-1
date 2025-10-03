from main import LinkedListDB, INPUT_CSV

OUTPUT_CSV_QUICK = "output_quick.csv"

def run_quick():
    db = LinkedListDB.load_from_csv_recursive(INPUT_CSV)
    db.sync_with_csv(INPUT_CSV)
    steps, dt = db.quick_sort(db.key_name, reverse=False)
    print(f"Quick sort: steps={steps}, seconds={dt:.6f}")
    db.export_to_csv_recursive(OUTPUT_CSV_QUICK)

if __name__ == "__main__":
    run_quick()
