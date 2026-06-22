"""
MANAJEMEN PENJADWALAN TUGAS DAN CEK PLAGIARISME
Menggunakan Red-Black Tree, Hash Table, Greedy, Linear Search, Rabin-Karp
Tanpa library eksternal - semua struktur data dibangun dari 0
"""

import csv
import os
from datetime import datetime, date

# ============================================================
# KONSTANTA
# ============================================================
FILE_TUGAS   = "tugas.csv"
FILE_KONTEN  = "konten.csv"
TABLE_SIZE   = 100       # ukuran Hash Table
WINDOW_SIZE  = 3         # ukuran window Rabin-Karp (k-gram)
RK_BASE      = 31        # base Rabin-Karp
RK_MOD       = 101       # modulus Rabin-Karp
ALARM_HARI   = 3         # batas hari alarm deadline
RED          = "RED"
BLACK        = "BLACK"

# Bobot kategori untuk Greedy
BOBOT_KATEGORI = {
    "individu" : 1.0,
    "kelompok" : 1.5,
}


# ============================================================
# CLASS TASK — Unit data utama
# ============================================================
class Task:
    """
    Representasi satu tugas mahasiswa.
    Semua struktur data (RBT, HashTable, CSV) beroperasi pada objek ini.
    """
    def __init__(self, id, nama_tugas, mata_kuliah, kategori, deadline, bobot=0.0):
        self.id          = int(id)
        self.nama_tugas  = nama_tugas.strip()
        self.mata_kuliah = mata_kuliah.strip()
        self.kategori    = kategori.strip().lower()   # "individu" / "kelompok"
        self.deadline    = deadline.strip()            # format "YYYY-MM-DD"
        self.bobot       = float(bobot)

    def __str__(self):
        return (f"[ID:{self.id}] {self.nama_tugas} | {self.mata_kuliah} | "
                f"{self.kategori.capitalize()} | Deadline: {self.deadline} | "
                f"Bobot: {self.bobot:.2f}")


# ============================================================
# GREEDY — Hitung bobot otomatis
# ============================================================
def hitung_bobot(kategori: str, deadline_str: str) -> float:
    """
    Greedy: hitung bobot tugas berdasarkan kategori dan sisa hari deadline.
    Rumus: bobot = bobot_kategori + (K / sisa_hari)
    Makin mepet deadline → bobot makin tinggi → makin diprioritaskan.
    
    K = 10.0 sebagai konstanta penguat urgensi.
    Jika deadline sudah lewat, sisa_hari = 0.1 (agar tidak ZeroDivision).
    """
    K = 10.0
    bobot_kat = BOBOT_KATEGORI.get(kategori.lower(), 1.0)
    try:
        tgl_deadline = datetime.strptime(deadline_str, "%Y-%m-%d").date()
        sisa_hari    = (tgl_deadline - date.today()).days
        if sisa_hari <= 0:
            sisa_hari = 0.1   # deadline lewat / hari ini = urgensi maksimal
    except ValueError:
        sisa_hari = 1.0
    bobot = bobot_kat + (K / sisa_hari)
    return round(bobot, 2)


# ============================================================
# RED-BLACK TREE
# ============================================================
class RBNode:
    """
    Node Red-Black Tree.
    Key = tuple (deadline_str, id) → deadline sebagai prioritas, id sebagai tiebreaker.
    """
    def __init__(self, task: Task):
        self.task   = task
        self.key    = (task.deadline, task.id)
        self.color  = RED
        self.left   = None
        self.right  = None
        self.parent = None


class RedBlackTree:
    """
    Implementasi Red-Black Tree dari nol.
    
    Properti RBT yang dijaga:
    1. Setiap node berwarna RED atau BLACK
    2. Root selalu BLACK
    3. Setiap leaf (NIL) adalah BLACK
    4. Node RED tidak boleh punya anak RED (no double red)
    5. Setiap path dari root ke leaf punya jumlah BLACK node yang sama
    
    Operasi: insert, delete, search, inorder traversal — semua O(log n)
    """

    def __init__(self):
        # NIL sentinel — semua leaf mengarah ke node ini
        self.NIL        = RBNode.__new__(RBNode)
        self.NIL.color  = BLACK
        self.NIL.left   = None
        self.NIL.right  = None
        self.NIL.parent = None
        self.NIL.task   = None
        self.NIL.key    = None
        self.root       = self.NIL

    # ── ROTASI ──────────────────────────────────────────────
    def _rotate_left(self, x):
        """Rotasi kiri: x turun ke kiri, kanan x (y) naik."""
        y         = x.right
        x.right   = y.left
        if y.left != self.NIL:
            y.left.parent = x
        y.parent  = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left  = y
        else:
            x.parent.right = y
        y.left    = x
        x.parent  = y

    def _rotate_right(self, x):
        """Rotasi kanan: x turun ke kanan, kiri x (y) naik."""
        y         = x.left
        x.left    = y.right
        if y.right != self.NIL:
            y.right.parent = x
        y.parent  = x.parent
        if x.parent is None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left  = y
        y.right   = x
        x.parent  = y

    # ── INSERT ──────────────────────────────────────────────
    def insert(self, task: Task):
        """Sisipkan task baru ke RBT, lalu fix pelanggaran properti."""
        node        = RBNode(task)
        node.left   = self.NIL
        node.right  = self.NIL

        # BST insert biasa
        parent  = None
        current = self.root
        while current != self.NIL:
            parent = current
            if node.key < current.key:
                current = current.left
            else:
                current = current.right

        node.parent = parent
        if parent is None:
            self.root = node
        elif node.key < parent.key:
            parent.left  = node
        else:
            parent.right = node

        # Node baru = RED; fix jika ada double red
        node.color = RED
        self._fix_insert(node)

    def _fix_insert(self, z):
        """Perbaiki properti RBT setelah insert."""
        while z.parent and z.parent.color == RED:
            if z.parent == z.parent.parent.left:
                y = z.parent.parent.right   # uncle
                if y.color == RED:
                    # Case 1: uncle RED → recolor
                    z.parent.color         = BLACK
                    y.color                = BLACK
                    z.parent.parent.color  = RED
                    z                      = z.parent.parent
                else:
                    if z == z.parent.right:
                        # Case 2: z adalah anak kanan → rotate kiri
                        z = z.parent
                        self._rotate_left(z)
                    # Case 3: z adalah anak kiri → rotate kanan
                    z.parent.color        = BLACK
                    z.parent.parent.color = RED
                    self._rotate_right(z.parent.parent)
            else:
                # Mirror dari kasus di atas
                y = z.parent.parent.left
                if y.color == RED:
                    z.parent.color        = BLACK
                    y.color               = BLACK
                    z.parent.parent.color = RED
                    z                     = z.parent.parent
                else:
                    if z == z.parent.left:
                        z = z.parent
                        self._rotate_right(z)
                    z.parent.color        = BLACK
                    z.parent.parent.color = RED
                    self._rotate_left(z.parent.parent)
        self.root.color = BLACK

    # ── DELETE ──────────────────────────────────────────────
    def delete(self, task_id: int, deadline_str: str):
        """Hapus node dengan key (deadline, id) dari RBT."""
        key  = (deadline_str, task_id)
        node = self._search_node(key)
        if node == self.NIL or node is None:
            return
        self._delete_node(node)

    def _search_node(self, key):
        current = self.root
        while current != self.NIL:
            if key == current.key:
                return current
            elif key < current.key:
                current = current.left
            else:
                current = current.right
        return self.NIL

    def _minimum(self, node):
        while node.left != self.NIL:
            node = node.left
        return node

    def _transplant(self, u, v):
        """Ganti subtree u dengan subtree v."""
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left  = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _delete_node(self, z):
        y              = z
        y_original_color = y.color
        if z.left == self.NIL:
            x = z.right
            self._transplant(z, z.right)
        elif z.right == self.NIL:
            x = z.left
            self._transplant(z, z.left)
        else:
            y              = self._minimum(z.right)
            y_original_color = y.color
            x              = y.right
            if y.parent == z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right        = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left         = z.left
            y.left.parent  = y
            y.color        = z.color
        if y_original_color == BLACK:
            self._fix_delete(x)

    def _fix_delete(self, x):
        """Perbaiki properti RBT setelah delete."""
        while x != self.root and x.color == BLACK:
            if x == x.parent.left:
                w = x.parent.right
                if w.color == RED:
                    w.color        = BLACK
                    x.parent.color = RED
                    self._rotate_left(x.parent)
                    w = x.parent.right
                if w.left.color == BLACK and w.right.color == BLACK:
                    w.color = RED
                    x       = x.parent
                else:
                    if w.right.color == BLACK:
                        w.left.color = BLACK
                        w.color      = RED
                        self._rotate_right(w)
                        w = x.parent.right
                    w.color        = x.parent.color
                    x.parent.color = BLACK
                    w.right.color  = BLACK
                    self._rotate_left(x.parent)
                    x = self.root
            else:
                w = x.parent.left
                if w.color == RED:
                    w.color        = BLACK
                    x.parent.color = RED
                    self._rotate_right(x.parent)
                    w = x.parent.left
                if w.right.color == BLACK and w.left.color == BLACK:
                    w.color = RED
                    x       = x.parent
                else:
                    if w.left.color == BLACK:
                        w.right.color = BLACK
                        w.color       = RED
                        self._rotate_left(w)
                        w = x.parent.left
                    w.color        = x.parent.color
                    x.parent.color = BLACK
                    w.left.color   = BLACK
                    self._rotate_right(x.parent)
                    x = self.root
        x.color = BLACK

    # ── TRAVERSAL ───────────────────────────────────────────
    def inorder(self) -> list:
        """In-order traversal → list Task terurut by deadline (ascending)."""
        result = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, node, result):
        if node != self.NIL:
            self._inorder_recursive(node.left, result)
            result.append(node.task)
            self._inorder_recursive(node.right, result)

    def is_empty(self) -> bool:
        return self.root == self.NIL


# ============================================================
# HASH TABLE — Storage lookup O(1)
# ============================================================
class HashNode:
    """Node untuk chaining di Hash Table."""
    def __init__(self, key, value):
        self.key   = key
        self.value = value
        self.next  = None


class HashTable:
    """
    Hash Table dengan chaining untuk collision handling.
    Key = task.id (int), Value = objek Task.
    Fungsi hash: h(id) = id % TABLE_SIZE
    """
    def __init__(self, size=TABLE_SIZE):
        self.size   = size
        self.table  = [None] * size   # array of linked list heads
        self.count  = 0

    def _hash(self, key: int) -> int:
        return int(key) % self.size

    def insert(self, task: Task):
        """Simpan task. Jika ID sudah ada, update nilainya."""
        idx  = self._hash(task.id)
        node = self.table[idx]
        # Cek apakah ID sudah ada (update)
        while node:
            if node.key == task.id:
                node.value = task
                return
            node = node.next
        # ID baru → tambah di depan linked list
        new_node       = HashNode(task.id, task)
        new_node.next  = self.table[idx]
        self.table[idx] = new_node
        self.count     += 1

    def get(self, task_id: int) -> Task:
        """Ambil task by ID. Return None jika tidak ada."""
        idx  = self._hash(task_id)
        node = self.table[idx]
        while node:
            if node.key == task_id:
                return node.value
            node = node.next
        return None

    def delete(self, task_id: int) -> bool:
        """Hapus task by ID. Return True jika berhasil."""
        idx  = self._hash(task_id)
        node = self.table[idx]
        prev = None
        while node:
            if node.key == task_id:
                if prev:
                    prev.next = node.next
                else:
                    self.table[idx] = node.next
                self.count -= 1
                return True
            prev = node
            node = node.next
        return False

    def get_all(self) -> list:
        """Ambil semua task dalam Hash Table."""
        result = []
        for node in self.table:
            current = node
            while current:
                result.append(current.value)
                current = current.next
        return result

    def get_next_id(self) -> int:
        """Generate ID baru = max ID yang ada + 1."""
        all_tasks = self.get_all()
        if not all_tasks:
            return 1
        return max(t.id for t in all_tasks) + 1


# ============================================================
# LINEAR SEARCH + ALARM DEADLINE
# ============================================================
def linear_search(hash_table: HashTable, keyword: str) -> list:
    """
    Cari tugas berdasarkan keyword (nama tugas atau mata kuliah).
    Sekaligus cek alarm jika deadline <= ALARM_HARI hari dari sekarang.
    Return: list of (task, sisa_hari, alarm_aktif)
    """
    keyword    = keyword.lower().strip()
    all_tasks  = hash_table.get_all()
    hasil      = []
    today      = date.today()

    for task in all_tasks:
        # Cek kecocokan keyword
        if keyword in task.nama_tugas.lower() or keyword in task.mata_kuliah.lower():
            try:
                tgl_deadline = datetime.strptime(task.deadline, "%Y-%m-%d").date()
                sisa_hari    = (tgl_deadline - today).days
            except ValueError:
                sisa_hari    = -1
            alarm_aktif  = 0 <= sisa_hari <= ALARM_HARI
            hasil.append((task, sisa_hari, alarm_aktif))

    return hasil


def cek_alarm_semua(hash_table: HashTable) -> list:
    """
    Cek semua tugas yang deadline-nya <= ALARM_HARI hari.
    Dipanggil saat program start untuk early warning.
    Return: list of (task, sisa_hari)
    """
    all_tasks = hash_table.get_all()
    today     = date.today()
    alarm     = []

    for task in all_tasks:
        try:
            tgl_deadline = datetime.strptime(task.deadline, "%Y-%m-%d").date()
            sisa_hari    = (tgl_deadline - today).days
            if 0 <= sisa_hari <= ALARM_HARI:
                alarm.append((task, sisa_hari))
        except ValueError:
            pass

    # Urutkan by sisa hari (makin mepet makin atas)
    alarm.sort(key=lambda x: x[1])
    return alarm


# ============================================================
# RABIN-KARP — Cek Plagiarisme
# ============================================================
def _preprocessing(teks: str) -> list:
    """
    Bersihkan teks: lowercase, hapus tanda baca, split jadi list kata.
    Tanpa library regex — manual character filtering.
    """
    teks  = teks.lower()
    bersih = ""
    for ch in teks:
        if ch.isalnum() or ch == " ":
            bersih += ch
        else:
            bersih += " "
    return [w for w in bersih.split() if w]


def _nilai_kata(kata: str) -> int:
    """Konversi kata ke nilai numerik (sum of ord tiap karakter)."""
    total = 0
    for ch in kata:
        total += ord(ch)
    return total


def rabin_karp(teks_a: str, teks_b: str, k: int = WINDOW_SIZE) -> dict:
    """
    Deteksi kemiripan dua teks menggunakan Rabin-Karp rolling hash.
    
    Langkah:
    1. Preprocessing kedua teks
    2. Sliding window k kata → hitung rolling hash tiap window
    3. Simpan semua hash ke Set_A dan Set_B
    4. Irisan Set_A ∩ Set_B = segmen yang mirip
    5. Kemiripan = |irisan| / max(|Set_A|, |Set_B|) × 100%
    
    Return: dict berisi persentase, jumlah segmen, segmen mirip
    """
    kata_a = _preprocessing(teks_a)
    kata_b = _preprocessing(teks_b)

    if len(kata_a) < k or len(kata_b) < k:
        return {
            "persen"        : 0.0,
            "total_a"       : len(kata_a),
            "total_b"       : len(kata_b),
            "segmen_mirip"  : [],
            "pesan"         : f"Teks terlalu pendek (minimal {k} kata)."
        }

    def buat_set_hash(kata_list):
        """Buat set hash dari semua window k kata (rolling hash)."""
        hash_set    = {}   # hash_val → list of window string (untuk tampil segmen mirip)
        n           = len(kata_list)

        # Hitung hash window pertama
        h = 0
        power = 1
        for i in range(k):
            nilai = _nilai_kata(kata_list[i])
            h    += nilai * power
            h    %= RK_MOD
            if i < k - 1:
                power = (power * RK_BASE) % RK_MOD

        window_str = " ".join(kata_list[0:k])
        if h not in hash_set:
            hash_set[h] = []
        hash_set[h].append(window_str)

        # Rolling hash untuk window berikutnya
        for i in range(1, n - k + 1):
            nilai_keluar = _nilai_kata(kata_list[i - 1])
            nilai_masuk  = _nilai_kata(kata_list[i + k - 1])
            h = (h - nilai_keluar) % RK_MOD
            h = (h * RK_BASE + nilai_masuk) % RK_MOD
            h = h % RK_MOD

            window_str = " ".join(kata_list[i:i + k])
            if h not in hash_set:
                hash_set[h] = []
            hash_set[h].append(window_str)

        return hash_set

    set_a = buat_set_hash(kata_a)
    set_b = buat_set_hash(kata_b)

    # Hitung irisan
    segmen_mirip = []
    irisan_count = 0
    for h_val in set_a:
        if h_val in set_b:
            irisan_count += 1
            segmen_mirip.extend(set_a[h_val])

    total_a = len(set_a)
    total_b = len(set_b)
    denom   = max(total_a, total_b)
    persen  = (irisan_count / denom * 100) if denom > 0 else 0.0

    return {
        "persen"       : round(persen, 2),
        "total_a"      : total_a,
        "total_b"      : total_b,
        "irisan"       : irisan_count,
        "segmen_mirip" : segmen_mirip[:10],   # tampilkan max 10 segmen
        "pesan"        : ""
    }


# ============================================================
# CSV MANAGER — Persistensi data
# ============================================================
class CSVManager:
    """
    Kelola baca/tulis file CSV.
    Menggunakan modul csv bawaan Python — bukan library eksternal.
    Auto-save dipanggil setiap kali ada perubahan data.
    """

    @staticmethod
    def simpan_tugas(hash_table: HashTable):
        """Tulis ulang tugas.csv dari seluruh isi Hash Table."""
        all_tasks = hash_table.get_all()
        with open(FILE_TUGAS, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "nama_tugas", "mata_kuliah", "kategori", "deadline", "bobot"])
            for t in all_tasks:
                writer.writerow([t.id, t.nama_tugas, t.mata_kuliah, t.kategori, t.deadline, t.bobot])

    @staticmethod
    def load_tugas(hash_table: HashTable, rbt: RedBlackTree):
        """Muat tugas.csv ke Hash Table dan RB Tree saat startup."""
        if not os.path.exists(FILE_TUGAS):
            return
        with open(FILE_TUGAS, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    task = Task(
                        id          = row["id"],
                        nama_tugas  = row["nama_tugas"],
                        mata_kuliah = row["mata_kuliah"],
                        kategori    = row["kategori"],
                        deadline    = row["deadline"],
                        bobot       = row["bobot"]
                    )
                    hash_table.insert(task)
                    rbt.insert(task)
                except (KeyError, ValueError):
                    continue

    @staticmethod
    def simpan_konten(konten_dict: dict):
        """
        Tulis ulang konten.csv.
        konten_dict = {id: {"judul": ..., "konten": ...}}
        """
        with open(FILE_KONTEN, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "judul", "konten_teks"])
            for kid, val in konten_dict.items():
                writer.writerow([kid, val["judul"], val["konten"]])

    @staticmethod
    def load_konten() -> dict:
        """Muat konten.csv → dict {id: {"judul":..., "konten":...}}"""
        hasil = {}
        if not os.path.exists(FILE_KONTEN):
            return hasil
        with open(FILE_KONTEN, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    hasil[int(row["id"])] = {
                        "judul"  : row["judul"],
                        "konten" : row["konten_teks"]
                    }
                except (KeyError, ValueError):
                    continue
        return hasil

    @staticmethod
    def get_next_konten_id(konten_dict: dict) -> int:
        """Generate ID konten baru = max ID + 1."""
        if not konten_dict:
            return 1
        return max(konten_dict.keys()) + 1


# ============================================================
# EXPORT .ICS — Google Calendar
# ============================================================
def export_ics(rbt: RedBlackTree, filename: str = "jadwal_tugas.ics"):
    """
    Export jadwal ke file iCalendar (.ics).
    Data diambil via in-order traversal RBT → sudah terurut by deadline.
    Format .ics adalah plain text — tidak butuh library apapun.
    """
    tasks = rbt.inorder()
    if not tasks:
        print("  Tidak ada tugas untuk diekspor.")
        return

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Sistem Tugas UNEJ//ID",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
    ]

    for task in tasks:
        # Konversi deadline "YYYY-MM-DD" ke format iCal "YYYYMMDD"
        tgl_ical = task.deadline.replace("-", "")
        uid      = f"task-{task.id}@sistemtugas.unej"
        lines += [
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"SUMMARY:{task.nama_tugas}",
            f"DTSTART;VALUE=DATE:{tgl_ical}",
            f"DTEND;VALUE=DATE:{tgl_ical}",
            f"DESCRIPTION:Mata Kuliah: {task.mata_kuliah} | Kategori: {task.kategori.capitalize()} | Bobot: {task.bobot}",
            "END:VEVENT",
        ]

    lines.append("END:VCALENDAR")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"  Jadwal berhasil diekspor ke '{filename}'")
    print(f"  Total {len(tasks)} tugas diekspor.")


# ============================================================
# UI HELPERS
# ============================================================
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def garis(char="─", panjang=60):
    print(char * panjang)

def header():
    clear_screen()
    garis("═")
    print("  SISTEM MANAJEMEN TUGAS & CEK PLAGIARISME")
    garis("═")

def tampil_menu():
    print("         MENU UTAMA")
    garis()
    print("  [1] Tambah Tugas")
    print("  [2] Lihat Jadwal")
    print("  [3] Cari Tugas")
    print("  [4] Hapus Tugas")
    print("  [5] Edit Tugas")
    print("  [6] Cek Plagiarisme")
    print("  [7] Ekspor ke .ics (Google Calendar)")
    print("  [0] Keluar")
    garis()

def tampil_task_row(task: Task, sisa_hari=None):
    """Tampilkan satu baris tugas, dengan warna alarm jika perlu."""
    alarm = ""
    if sisa_hari is not None and 0 <= sisa_hari <= ALARM_HARI:
        alarm = f"  ⚠ DEADLINE {sisa_hari} HARI LAGI!"
    elif sisa_hari is not None and sisa_hari < 0:
        alarm = "  ✗ DEADLINE LEWAT"
    print(f"  {task}{alarm}")


# ============================================================
# MENU 1 — TAMBAH TUGAS
# ============================================================
def menu_tambah(hash_table: HashTable, rbt: RedBlackTree):
    header()
    print("  TAMBAH TUGAS BARU\n")
    nama       = input("  Nama Tugas      : ").strip()
    matkul     = input("  Mata Kuliah     : ").strip()
    print("  Kategori        : [1] Individu  [2] Kelompok")
    kat_input  = input("  Pilih           : ").strip()
    kategori   = "individu" if kat_input == "1" else "kelompok"

    while True:
        deadline = input("  Deadline (YYYY-MM-DD): ").strip()
        try:
            datetime.strptime(deadline, "%Y-%m-%d")
            break
        except ValueError:
            print("  Format salah! Gunakan YYYY-MM-DD.")

    if not nama or not matkul:
        print("  Nama tugas dan mata kuliah tidak boleh kosong!")
        input("  Enter untuk kembali...")
        return

    task_id = hash_table.get_next_id()
    bobot   = hitung_bobot(kategori, deadline)
    task    = Task(task_id, nama, matkul, kategori, deadline, bobot)

    hash_table.insert(task)
    rbt.insert(task)
    CSVManager.simpan_tugas(hash_table)

    print(f"\n  ✓ Tugas berhasil ditambahkan!")
    print(f"  ID: {task_id} | Bobot otomatis: {bobot}")
    input("\n  Enter untuk kembali...")


# ============================================================
# MENU 2 — LIHAT JADWAL
# ============================================================
def menu_lihat_jadwal(rbt: RedBlackTree):
    header()
    print("  JADWAL TUGAS (urut by deadline)\n")
    tasks = rbt.inorder()
    if not tasks:
        print("  Belum ada tugas.")
        input("\n  Enter untuk kembali...")
        return

    today = date.today()
    garis()
    for task in tasks:
        try:
            tgl  = datetime.strptime(task.deadline, "%Y-%m-%d").date()
            sisa = (tgl - today).days
        except ValueError:
            sisa = None
        tampil_task_row(task, sisa)
    garis()
    print(f"\n  Total: {len(tasks)} tugas")
    input("\n  Enter untuk kembali...")


# ============================================================
# MENU 3 — CARI TUGAS
# ============================================================
def menu_cari(hash_table: HashTable):
    header()
    print("  CARI TUGAS\n")
    keyword = input("  Keyword (nama tugas / mata kuliah): ").strip()
    if not keyword:
        input("  Keyword kosong. Enter untuk kembali...")
        return

    hasil = linear_search(hash_table, keyword)
    print()
    garis()
    if not hasil:
        print(f"  Tidak ada tugas yang cocok dengan '{keyword}'.")
    else:
        print(f"  Ditemukan {len(hasil)} tugas:\n")
        for task, sisa_hari, alarm in hasil:
            tampil_task_row(task, sisa_hari)
    garis()
    input("\n  Enter untuk kembali...")


# ============================================================
# MENU 4 — HAPUS TUGAS
# ============================================================
def menu_hapus(hash_table: HashTable, rbt: RedBlackTree):
    header()
    print("  HAPUS TUGAS\n")

    # Tampilkan semua tugas dulu
    tasks = rbt.inorder()
    if not tasks:
        print("  Belum ada tugas.")
        input("\n  Enter untuk kembali...")
        return

    for task in tasks:
        print(f"  {task}")

    garis()
    try:
        task_id = int(input("\n  Masukkan ID tugas yang akan dihapus: ").strip())
    except ValueError:
        print("  ID tidak valid.")
        input("  Enter untuk kembali...")
        return

    task = hash_table.get(task_id)
    if not task:
        print(f"  Tugas dengan ID {task_id} tidak ditemukan.")
        input("  Enter untuk kembali...")
        return

    konfirmasi = input(f"  Hapus '{task.nama_tugas}'? (y/n): ").strip().lower()
    if konfirmasi != "y":
        print("  Dibatalkan.")
        input("  Enter untuk kembali...")
        return

    rbt.delete(task_id, task.deadline)
    hash_table.delete(task_id)
    CSVManager.simpan_tugas(hash_table)

    print(f"\n  ✓ Tugas '{task.nama_tugas}' berhasil dihapus.")
    input("\n  Enter untuk kembali...")


# ============================================================
# MENU 5 — EDIT TUGAS
# ============================================================
def menu_edit(hash_table: HashTable, rbt: RedBlackTree):
    header()
    print("  EDIT TUGAS\n")

    tasks = rbt.inorder()
    if not tasks:
        print("  Belum ada tugas.")
        input("\n  Enter untuk kembali...")
        return

    for task in tasks:
        print(f"  {task}")

    garis()
    try:
        task_id = int(input("\n  Masukkan ID tugas yang akan diedit: ").strip())
    except ValueError:
        print("  ID tidak valid.")
        input("  Enter untuk kembali...")
        return

    task = hash_table.get(task_id)
    if not task:
        print(f"  Tugas ID {task_id} tidak ditemukan.")
        input("  Enter untuk kembali...")
        return

    print(f"\n  Edit tugas: {task.nama_tugas}")
    print("  (Kosongkan untuk mempertahankan nilai lama)\n")

    nama_baru   = input(f"  Nama Tugas [{task.nama_tugas}]: ").strip()
    matkul_baru = input(f"  Mata Kuliah [{task.mata_kuliah}]: ").strip()
    print(f"  Kategori [{task.kategori}]: [1] Individu  [2] Kelompok  [Enter] skip")
    kat_input   = input("  Pilih: ").strip()
    deadline_baru = ""
    while True:
        deadline_baru = input(f"  Deadline [{task.deadline}]: ").strip()
        if not deadline_baru:
            deadline_baru = task.deadline
            break
        try:
            datetime.strptime(deadline_baru, "%Y-%m-%d")
            break
        except ValueError:
            print("  Format salah! Gunakan YYYY-MM-DD.")

    # Ambil nilai baru atau tetap nilai lama
    nama_final   = nama_baru   if nama_baru   else task.nama_tugas
    matkul_final = matkul_baru if matkul_baru else task.mata_kuliah
    if kat_input == "1":
        kat_final = "individu"
    elif kat_input == "2":
        kat_final = "kelompok"
    else:
        kat_final = task.kategori

    # Hapus node lama dari RBT (key berdasar deadline lama)
    rbt.delete(task_id, task.deadline)

    # Hitung bobot baru
    bobot_baru = hitung_bobot(kat_final, deadline_baru)

    # Update task
    task.nama_tugas  = nama_final
    task.mata_kuliah = matkul_final
    task.kategori    = kat_final
    task.deadline    = deadline_baru
    task.bobot       = bobot_baru

    # Re-insert ke RBT dengan deadline baru, update Hash Table
    rbt.insert(task)
    hash_table.insert(task)
    CSVManager.simpan_tugas(hash_table)

    print(f"\n  ✓ Tugas berhasil diperbarui! Bobot baru: {bobot_baru}")
    input("\n  Enter untuk kembali...")


# ============================================================
# MENU 6 — CEK PLAGIARISME (submenu)
# ============================================================
def menu_plagiarisme(konten_dict: dict):
    while True:
        header()
        print("  CEK PLAGIARISME\n")
        print("  [1] Input & Simpan Teks Baru")
        print("  [2] Cek Plagiarisme Teks Tersimpan")
        print("  [3] Cek Plagiarisme Teks Bebas (tidak disimpan)")
        print("  [0] Kembali")
        garis()
        pilihan = input("  Pilih: ").strip()

        if pilihan == "1":
            _plagiarisme_input_simpan(konten_dict)
        elif pilihan == "2":
            _plagiarisme_cek_tersimpan(konten_dict)
        elif pilihan == "3":
            _plagiarisme_cek_bebas()
        elif pilihan == "0":
            break
        else:
            print("  Pilihan tidak valid.")
            input("  Enter untuk lanjut...")


def _plagiarisme_input_simpan(konten_dict: dict):
    """Submenu 6.1 — Input teks baru dan simpan ke konten.csv."""
    header()
    print("  INPUT TEKS BARU\n")
    judul  = input("  Judul / Label Teks: ").strip()
    if not judul:
        print("  Judul tidak boleh kosong.")
        input("  Enter untuk kembali...")
        return
    print("  Masukkan teks (ketik 'SELESAI' di baris baru untuk mengakhiri):")
    baris = []
    while True:
        line = input()
        if line.strip().upper() == "SELESAI":
            break
        baris.append(line)
    konten = " ".join(baris).strip()
    if not konten:
        print("  Konten kosong, tidak disimpan.")
        input("  Enter untuk kembali...")
        return

    new_id = CSVManager.get_next_konten_id(konten_dict)
    konten_dict[new_id] = {"judul": judul, "konten": konten}
    CSVManager.simpan_konten(konten_dict)
    print(f"\n  ✓ Teks disimpan dengan ID {new_id}.")
    input("\n  Enter untuk kembali...")


def _plagiarisme_cek_tersimpan(konten_dict: dict):
    """Submenu 6.2 — Pilih 2 teks dari konten.csv untuk dicek."""
    header()
    print("  CEK PLAGIARISME TEKS TERSIMPAN\n")

    if len(konten_dict) < 2:
        print("  Minimal 2 teks tersimpan untuk melakukan pengecekan.")
        input("  Enter untuk kembali...")
        return

    # Tampilkan daftar teks
    garis()
    for kid, val in konten_dict.items():
        preview = val["konten"][:60] + "..." if len(val["konten"]) > 60 else val["konten"]
        print(f"  [ID:{kid}] {val['judul']} — {preview}")
    garis()

    try:
        id_a = int(input("\n  Pilih ID Teks A: ").strip())
        id_b = int(input("  Pilih ID Teks B: ").strip())
    except ValueError:
        print("  ID tidak valid.")
        input("  Enter untuk kembali...")
        return

    if id_a == id_b:
        print("  Tidak bisa membandingkan teks yang sama.")
        input("  Enter untuk kembali...")
        return

    if id_a not in konten_dict or id_b not in konten_dict:
        print("  ID tidak ditemukan.")
        input("  Enter untuk kembali...")
        return

    _tampil_hasil_plagiarisme(
        konten_dict[id_a]["konten"], konten_dict[id_b]["konten"],
        konten_dict[id_a]["judul"],  konten_dict[id_b]["judul"]
    )


def _plagiarisme_cek_bebas():
    """Submenu 6.3 — Input 2 teks bebas, cek in-memory, tidak disimpan."""
    header()
    print("  CEK PLAGIARISME TEKS BEBAS\n")
    print("  Teks A (ketik 'SELESAI' untuk mengakhiri):")
    baris_a = []
    while True:
        line = input()
        if line.strip().upper() == "SELESAI":
            break
        baris_a.append(line)
    teks_a = " ".join(baris_a).strip()

    print("\n  Teks B (ketik 'SELESAI' untuk mengakhiri):")
    baris_b = []
    while True:
        line = input()
        if line.strip().upper() == "SELESAI":
            break
        baris_b.append(line)
    teks_b = " ".join(baris_b).strip()

    _tampil_hasil_plagiarisme(teks_a, teks_b, "Teks A", "Teks B")


def _tampil_hasil_plagiarisme(teks_a, teks_b, label_a="A", label_b="B"):
    """Jalankan Rabin-Karp dan tampilkan hasil."""
    hasil = rabin_karp(teks_a, teks_b)
    print()
    garis("═")
    print("  HASIL CEK PLAGIARISME")
    garis("═")

    if hasil["pesan"]:
        print(f"  ⚠ {hasil['pesan']}")
    else:
        persen = hasil["persen"]
        print(f"  Teks A : {label_a} ({hasil['total_a']} segmen)")
        print(f"  Teks B : {label_b} ({hasil['total_b']} segmen)")
        print(f"  Irisan : {hasil['irisan']} segmen cocok")
        garis()
        print(f"  KEMIRIPAN : {persen:.2f}%")

        # Interpretasi
        if persen >= 80:
            print("  Status    : ⛔ SANGAT MIRIP — Indikasi plagiarisme kuat")
        elif persen >= 50:
            print("  Status    : ⚠ CUKUP MIRIP — Perlu diperiksa lebih lanjut")
        elif persen >= 20:
            print("  Status    : ℹ SEDIKIT MIRIP — Kemungkinan kebetulan")
        else:
            print("  Status    : ✓ TIDAK MIRIP — Teks cukup original")

        if hasil["segmen_mirip"]:
            garis()
            print("  Segmen yang cocok (maks 10):")
            for i, seg in enumerate(hasil["segmen_mirip"], 1):
                print(f"    {i}. \"{seg}\"")

    garis("═")
    input("\n  Enter untuk kembali...")


# ============================================================
# MENU 7 — EKSPOR .ICS
# ============================================================
def menu_ekspor(rbt: RedBlackTree):
    header()
    print("  EKSPOR JADWAL KE .ICS\n")
    filename = input("  Nama file (default: jadwal_tugas.ics): ").strip()
    if not filename:
        filename = "jadwal_tugas.ics"
    if not filename.endswith(".ics"):
        filename += ".ics"
    export_ics(rbt, filename)
    input("\n  Enter untuk kembali...")


# ============================================================
# MAIN — Entry point
# ============================================================
def main():
    # Inisialisasi struktur data
    hash_table  = HashTable()
    rbt         = RedBlackTree()
    konten_dict = {}

    # Load data dari CSV saat startup
    CSVManager.load_tugas(hash_table, rbt)
    konten_dict = CSVManager.load_konten()

    # Cek alarm saat startup
    header()
    alarm_list = cek_alarm_semua(hash_table)
    if alarm_list:
        print(f"\n  ⚠  PERINGATAN DEADLINE MEPET ({len(alarm_list)} tugas):\n")
        for task, sisa in alarm_list:
            ket = "HARI INI!" if sisa == 0 else f"{sisa} hari lagi"
            print(f"  → {task.nama_tugas} ({task.mata_kuliah}) — {ket}")
        print()
        input("  Enter untuk melanjutkan ke menu...")

    # Loop menu utama
    while True:
        header()
        tampil_menu()
        pilihan = input("  Pilih menu: ").strip()

        if pilihan == "1":
            menu_tambah(hash_table, rbt)
        elif pilihan == "2":
            menu_lihat_jadwal(rbt)
        elif pilihan == "3":
            menu_cari(hash_table)
        elif pilihan == "4":
            menu_hapus(hash_table, rbt)
        elif pilihan == "5":
            menu_edit(hash_table, rbt)
        elif pilihan == "6":
            menu_plagiarisme(konten_dict)
        elif pilihan == "7":
            menu_ekspor(rbt)
        elif pilihan == "0":
            print("\n  Sampai jumpa! Data tersimpan otomatis.\n")
            break
        else:
            print("  Pilihan tidak valid.")
            input("  Enter untuk lanjut...")


if __name__ == "__main__":
    main()