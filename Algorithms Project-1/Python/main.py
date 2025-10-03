import csv
import time

INPUT_CSV = "student-data.csv"
OUTPUT_CSV_BUBBLE = "output_bubble.csv"
OUTPUT_CSV_INSERTION = "output_insertion.csv"

class Node:
    def __init__(self, record, key_name):
        self.record = record
        self.key_name = key_name
        self.key = record[key_name]
        self.next = None

class LinkedListDB:
    def __init__(self, key_name, headers):
        self.head = None
        self.tail = None
        self.size = 0
        self.key_name = key_name
        self.headers = headers

    def append(self, record):
        node = Node(record, self.key_name)
        if self.tail is None:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node
        self.size += 1

    def clear(self):
        self.head = None
        self.tail = None
        self.size = 0

    @classmethod
    def load_from_csv_recursive(cls, csv_filename):
        with open(csv_filename, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        headers = list(rows[0].keys())
        key_name = headers[0]
        db = cls(key_name, headers)
        def rec(i):
            if i == len(rows):
                return
            db.append(rows[i])
            rec(i + 1)
        rec(0)
        return db

    def sync_with_csv(self, csv_filename):
        with open(csv_filename, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        headers = list(rows[0].keys())
        self.headers = headers
        self.key_name = headers[0]
        self.clear()
        def rec(i):
            if i == len(rows):
                return
            self.append(rows[i])
            rec(i + 1)
        rec(0)

    def export_to_csv_recursive(self, csv_filename):
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.headers)
            w.writeheader()
            def rec(node):
                if node is None:
                    return
                w.writerow(node.record)
                rec(node.next)
            rec(self.head)

    def _val(self, rec, field):
        v = rec.get(field, "")
        try:
            return float(v)
        except:
            return str(v).lower()

    def bubble_sort(self, field, reverse=False):
        steps = 0
        if self.head is None or self.head.next is None:
            return steps, 0.0
        t0 = time.perf_counter()
        swapped = True
        while swapped:
            swapped = False
            cur = self.head
            while cur and cur.next:
                a = self._val(cur.record, field); b = self._val(cur.next.record, field)
                steps += 1
                cond = a > b if not reverse else a < b
                if cond:
                    cur.record, cur.next.record = cur.next.record, cur.record
                    steps += 1
                    swapped = True
                cur = cur.next
        dt = time.perf_counter() - t0
        return steps, dt

    def insertion_sort(self, field, reverse=False):
        steps = 0
        if self.head is None or self.head.next is None:
            return steps, 0.0
        t0 = time.perf_counter()
        sorted_head = None
        cur = self.head
        while cur:
            nxt = cur.next
            if sorted_head is None:
                cur.next = None
                sorted_head = cur
                steps += 1
            else:
                steps += 1
                if (self._val(cur.record, field) < self._val(sorted_head.record, field)) ^ reverse:
                    cur.next = sorted_head
                    sorted_head = cur
                    steps += 2
                else:
                    p = sorted_head
                    while p.next:
                        steps += 1
                        if ((self._val(p.next.record, field) <= self._val(cur.record, field)) ^ reverse) is False:
                            break
                        p = p.next
                    cur.next = p.next
                    p.next = cur
                    steps += 2
            cur = nxt
        self.head = sorted_head
        t = self.head
        while t and t.next:
            t = t.next
        self.tail = t
        dt = time.perf_counter() - t0
        return steps, dt

    def _val(self, rec, field):
        v = rec.get(field, "")
        try:
            return float(v)
        except:
            return str(v).lower()

    def _to_list(self):
        out = []
        n = self.head
        while n:
            out.append(n.record)
            n = n.next
        return out

    def _rebuild_from_list(self, records):
        self.clear()
        for r in records:
            self.append(r)

    def merge_sort(self, field, reverse=False):
        steps = 0
        t0 = time.perf_counter()
        arr = self._to_list()

        def cmp(a, b):
            A = self._val(a, field); B = self._val(b, field)
            c = (A > B) - (A < B)
            return -c if reverse else c

        def merge(left, right):
            nonlocal steps
            merged = []
            i = j = 0
            while i < len(left) and j < len(right):
                steps += 1
                if cmp(left[i], right[j]) <= 0:
                    merged.append(left[i]); i += 1
                else:
                    merged.append(right[j]); j += 1
            if i < len(left): merged.extend(left[i:])
            if j < len(right): merged.extend(right[j:])
            return merged

        def msort(a):
            if len(a) <= 1: return a
            mid = len(a)//2
            left = msort(a[:mid])
            right = msort(a[mid:])
            return merge(left, right)

        sorted_arr = msort(arr)
        self._rebuild_from_list(sorted_arr)
        dt = time.perf_counter() - t0
        return steps, dt

    def quick_sort(self, field, reverse=False):
        steps = 0
        t0 = time.perf_counter()
        arr = self._to_list()

        def cmp(a, b):
            A = self._val(a, field); B = self._val(b, field)
            c = (A > B) - (A < B)
            return -c if reverse else c

        def qsort(a):
            nonlocal steps
            if len(a) <= 1: return a
            pivot = a[len(a)//2]
            less, equal, greater = [], [], []
            for x in a:
                steps += 1
                c = cmp(x, pivot)
                if c < 0: less.append(x)
                elif c > 0: greater.append(x)
                else: equal.append(x)
            return qsort(less) + equal + qsort(greater)

        sorted_arr = qsort(arr)
        self._rebuild_from_list(sorted_arr)
        dt = time.perf_counter() - t0
        return steps, dt
