import openpyxl
import pandas as pd
import re
from pathlib import Path

class 胎儿:
    def __init__(self, 孕妇代码=None, 孕妇BMI=None, 检测孕周=None, 孕周_数值=None,检测抽血次数=None,
                 X染色体浓度=None, Y染色体浓度=None, 身高=None, 体重=None, GC含量=None,
                 胎儿是否健康=None, X染色体浓度集=None, 年龄=None,
                 染色体13_z值=None, 染色体18_z值=None, 染色体21_z值=None, 染色体X_z值=None,胎儿分数=None,
                 染色体的非整倍体=None):
        self.孕妇代码 = 孕妇代码
        self.孕妇BMI = 孕妇BMI
        self.检测孕周 = 检测孕周  # 保留原始格式
        self.孕周_数值 = self._转换孕周格式(检测孕周)  # 转换为数值
        self.检测抽血次数 = 检测抽血次数
        self.X染色体浓度 = X染色体浓度
        self.Y染色体浓度 = Y染色体浓度
        self.身高 = 身高
        self.体重 = 体重
        self.GC含量 = GC含量
        self.年龄 = 年龄
        self.胎儿是否健康 = 胎儿是否健康
        self.X染色体浓度集 = self._解析浓度集(X染色体浓度集)
        self.染色体13_z值 = 染色体13_z值
        self.染色体18_z值 = 染色体18_z值
        self.染色体21_z值 = 染色体21_z值
        self.染色体X_z值 = 染色体X_z值
        self.胎儿分数 = 胎儿分数
        self.染色体的非整倍体 = 染色体的非整倍体
    
    def _转换孕周格式(self, 孕周_str):
        孕周_str = str(孕周_str).strip()
        match = re.match(r'(\d+)w(?:\+(\d+))?', 孕周_str, re.IGNORECASE)
        if match:
            weeks = int(match.group(1))
            days = int(match.group(2)) if match.group(2) else 0
            return weeks + days / 7

    def _解析浓度集(self, v):
        if v is None:
            return None
        # 已经是列表/元组
        if isinstance(v, (list, tuple)):
            out = []
            for x in v:
                try:
                    out.append(float(x))
                except Exception:
                    continue
            return out if out else None
        # 尝试按常见分隔符拆分
        s = str(v).strip()
        if not s:
            return None
        for sep in [';', ',', '|', ' ', '，', '；']:
            s = s.replace(sep, ',')
        parts = [p for p in s.split(',') if p]
        out = []
        for p in parts:
            try:
                out.append(float(p))
            except Exception:
                continue
        return out if out else None

    def to_dict(self) -> dict:
        return dict(self.__dict__)
#清洗数据
def build(sheet_index: int = 0):
        # 读取数据（优先 数据导入/data.xlsx，不存在则回退 附件.xlsx）；支持选择工作表索引（0/1）
    base_dir = Path(__file__).resolve().parent
    cand1 = base_dir / "data.xlsx"
    cand2 = base_dir.parent / "数据导入" / "data.xlsx"
    cand3 = base_dir.parent / "3. 数据集" / "原始数据" / "附件.xlsx"
    src = cand1 if cand1.exists() else (cand2 if cand2.exists() else cand3)
    try:
        df = pd.read_excel(src, sheet_name=sheet_index)
    except Exception:
        # 兼容旧表或仅有单表的情况
        df = pd.read_excel(src)

    # 构造对象列表
    胎儿列表 = []
    for index, row in df.iterrows():
        obj = 胎儿(
            孕妇代码=row.get('孕妇代码', None),
            孕妇BMI=row.get('孕妇BMI', None),
            年龄=row.get('年龄', None),
            检测孕周=row.get('检测孕周', None),
            检测抽血次数=row.get('检测抽血次数', None),
            X染色体浓度=row.get('X染色体浓度', None),
            Y染色体浓度=row.get('Y染色体浓度', None),
            身高=row.get('身高', None),
            体重=row.get('体重', None),
            GC含量=row.get('GC含量', None),
            胎儿是否健康=row.get('胎儿是否健康', None),
            染色体13_z值=row.get('染色体13_z值', None),
            染色体18_z值=row.get('染色体18_z值', None),
            染色体21_z值=row.get('染色体21_z值', None),
            染色体X_z值=row.get('染色体X_z值', None),
            染色体的非整倍体=row.get('染色体的非整倍体', None)
        )
        胎儿列表.append(obj)

    # print(f"共创建了 {len(胎儿列表)} 个胎儿对象")

    # Step 1: 过滤 Y染色体浓度 >= 0.04
    step_1_objs = [
        t for t in 胎儿列表
        if (t.Y染色体浓度 is not None) and (t.Y染色体浓度 >= 0.04)
    ]
    # print(f"Y染色体浓度 >= 0.04 的胎儿数量: {len(step_1_objs)}")

    # Step 2: 过滤 孕周在 [10, 25]
    step_2_objs = [
        t for t in step_1_objs
        if (t.孕周_数值 is not None) and (10 <= t.孕周_数值 <= 25)
    ]
    # print(f"再筛孕周[10,25]的数量: {len(step_2_objs)}")

    # Step 3: 过滤 GC含量 >= 0.395
    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return None

    step_3_objs = []
    for t in step_2_objs:
        cg = _to_float(t.GC含量)
        if (cg is not None) and (cg >= 0.395):
            step_3_objs.append(t)
    # print(f"再筛GC含量>=0.395的数量: {len(step_3_objs)}")

    # Step 4: 对相同孕妇代码，选择“检测抽血次数”最大；若相同，再比较“GC含量”最大
    def _to_int(v):
        try:
            return int(float(v))
        except Exception:
            return -1
    def _score(t):
        """比较用:先看抽血次数,再看GC含量,均为越大越好"""
        gc_val = _to_float(t.GC含量)
        return (_to_int(t.检测抽血次数), gc_val if gc_val is not None else -1.0)

    code_to_best = {}
    for t in step_3_objs:
        code = t.孕妇代码
        if code is None:
            continue
        if (code not in code_to_best) or (_score(t) > _score(code_to_best[code])):
            code_to_best[code] = t

    step_4_objs = list(code_to_best.values())
    return step_4_objs
#一点不洗
def build2(sheet_index: int = 0):
    # 读取数据（优先 数据导入/data.xlsx，不存在则回退 附件.xlsx）；支持选择工作表索引
    base_dir = Path(__file__).resolve().parent
    cand1 = base_dir / "data.xlsx"
    cand2 = base_dir.parent / "数据导入" / "data.xlsx"
    cand3 = base_dir.parent / "3. 数据集" / "原始数据" / "附件.xlsx"
    src = cand1 if cand1.exists() else (cand2 if cand2.exists() else cand3)
    try:
        df = pd.read_excel(src, sheet_name=sheet_index)
    except Exception:
        df = pd.read_excel(src)

    # 构造对象列表（不做清洗与筛选）
    胎儿列表 = []
    for index, row in df.iterrows():
        obj = 胎儿(
            孕妇代码=row.get('孕妇代码', None),
            孕妇BMI=row.get('孕妇BMI', None),
            年龄=row.get('年龄', None),
            检测孕周=row.get('检测孕周', None),
            检测抽血次数=row.get('检测抽血次数', None),
            X染色体浓度=row.get('X染色体浓度', None),
            Y染色体浓度=row.get('Y染色体浓度', None),
            身高=row.get('身高', None),
            体重=row.get('体重', None),
            GC含量=row.get('GC含量', None),
            胎儿是否健康=row.get('胎儿是否健康', None),
            染色体13_z值=row.get('染色体13_z值', None),
            染色体18_z值=row.get('染色体18_z值', None),
            染色体21_z值=row.get('染色体21_z值', None),
            染色体X_z值=row.get('染色体X_z值', None),
            染色体的非整倍体=row.get('染色体的非整倍体', None)
        )
        胎儿列表.append(obj)
    return 胎儿列表
#洗一点
def build3(sheet_index: int = 0):
    # 读取数据（优先 数据导入/data.xlsx，不存在则回退 附件.xlsx）；支持选择工作表索引（部分清洗）
    base_dir = Path(__file__).resolve().parent
    cand1 = base_dir / "data.xlsx"
    cand2 = base_dir.parent / "数据导入" / "data.xlsx"
    cand3 = base_dir.parent / "3. 数据集" / "原始数据" / "附件.xlsx"
    src = cand1 if cand1.exists() else (cand2 if cand2.exists() else cand3)
    try:
        df = pd.read_excel(src, sheet_name=sheet_index)
    except Exception:
        df = pd.read_excel(src)

    # 构造对象列表
    胎儿列表 = []
    for index, row in df.iterrows():
        obj = 胎儿(
            孕妇代码=row.get('孕妇代码', None),
            孕妇BMI=row.get('孕妇BMI', None),
            年龄=row.get('年龄', None),
            检测孕周=row.get('检测孕周', None),
            检测抽血次数=row.get('检测抽血次数', None),
            X染色体浓度=row.get('X染色体浓度', None),
            Y染色体浓度=row.get('Y染色体浓度', None),
            身高=row.get('身高', None),
            体重=row.get('体重', None),
            GC含量=row.get('GC含量', None),
            胎儿是否健康=row.get('胎儿是否健康', None),
            染色体13_z值=row.get('染色体13_z值', None),
            染色体18_z值=row.get('染色体18_z值', None),
            染色体21_z值=row.get('染色体21_z值', None),
            染色体X_z值=row.get('染色体X_z值', None),
            染色体的非整倍体=row.get('染色体的非整倍体', None)
        )
        胎儿列表.append(obj)

    # Step 1: 过滤 Y染色体浓度 >= 0.04（保持与其他流程一致）
    step_1_objs = [
        t for t in 胎儿列表
        if (t.Y染色体浓度 is not None) and (t.Y染色体浓度 >= 0.04)
    ]

    # Step 2: 过滤 孕周在 [10, 25]
    step_2_objs = [
        t for t in step_1_objs
        if (t.孕周_数值 is not None) and (10 <= t.孕周_数值 <= 25)
    ]

    # Step 3: 对相同孕妇代码，选择“检测抽血次数”最大；若相同，再比较“GC含量”最大
    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return None
    def _to_int(v):
        try:
            return int(float(v))
        except Exception:
            return -1
    def _score(t):
        """比较用:先看抽血次数,再看GC含量,均为越大越好"""
        gc_val = _to_float(t.GC含量)
        return (_to_int(t.检测抽血次数), gc_val if gc_val is not None else -1.0)

    code_to_best = {}
    for t in step_2_objs:
        code = t.孕妇代码
        if code is None:
            continue
        if (code not in code_to_best) or (_score(t) > _score(code_to_best[code])):
            code_to_best[code] = t

    return list(code_to_best.values())
        
if __name__ == "__main__":
    # 读取数据（优先 数据导入/data.xlsx，不存在则回退 附件.xlsx）
    base_dir = Path(__file__).resolve().parent
    cand1 = base_dir / "data.xlsx"
    cand2 = base_dir.parent / "数据导入" / "data.xlsx"
    cand3 = base_dir.parent / "3. 数据集" / "原始数据" / "附件.xlsx"
    src = cand1 if cand1.exists() else (cand2 if cand2.exists() else cand3)
    df = pd.read_excel(src)

    # 构造对象列表
    胎儿列表 = []
    for index, row in df.iterrows():
        obj = 胎儿(
            孕妇代码=row.get('孕妇代码', None),
            孕妇BMI=row.get('孕妇BMI', None),
            检测孕周=row.get('检测孕周', None),
            孕周_数值=row.get('孕周_数值', None),
            检测抽血次数=row.get('检测抽血次数', None),
            X染色体浓度=row.get('X染色体浓度', None),
            Y染色体浓度=row.get('Y染色体浓度', None),
            身高=row.get('身高', None),
            体重=row.get('体重', None),
            GC含量=row.get('GC含量', None)
        )
        胎儿列表.append(obj)

    print(f"共创建了 {len(胎儿列表)} 个胎儿对象")

    # Step 1: 过滤 Y染色体浓度 >= 0.04
    step_1_objs = [
        t for t in 胎儿列表
        if (t.Y染色体浓度 is not None) and (t.Y染色体浓度 >= 0.04)
    ]
    print(f"Y染色体浓度 >= 0.04 的胎儿数量: {len(step_1_objs)}")

    # Step 2: 过滤 孕周在 [10, 25]
    step_2_objs = [
        t for t in step_1_objs
        if (t.孕周_数值 is not None) and (10 <= t.孕周_数值 <= 25)
    ]
    print(f"再筛孕周[10,25]的数量: {len(step_2_objs)}")

    # Step 3: 过滤 GC含量 >= 0.41
    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return None

    step_3_objs = []
    for t in step_2_objs:
        cg = _to_float(t.GC含量)
        if (cg is not None) and (cg >= 0.40):
            step_3_objs.append(t)
    print(f"再筛GC含量>=0.40的数量: {len(step_3_objs)}")

    # Step 4: 对相同孕妇代码，选择“检测抽血次数”最大；若相同，再比较“GC含量”最大
    def _to_int(v):
        try:
            return int(float(v))
        except Exception:
            return -1
    def _score(t):
        """比较用:先看抽血次数,再看GC含量,均为越大越好"""
        gc_val = _to_float(t.GC含量)
        return (_to_int(t.检测抽血次数), gc_val if gc_val is not None else -1.0)

    code_to_best = {}
    for t in step_3_objs:
        code = t.孕妇代码
        if code is None:
            continue
        if (code not in code_to_best) or (_score(t) > _score(code_to_best[code])):
            code_to_best[code] = t

    step_4_objs = list(code_to_best.values())
    print(f"按最大检测抽血次数去重（并以GC含量打破并列）后的数量: {len(step_4_objs)}")

    for i, t in enumerate(step_4_objs[:3]):
        print(t.to_dict())