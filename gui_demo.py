import tkinter as tk
from tkinter import ttk, messagebox
import json
import os

class PersonalInfoManager:
    def __init__(self, root):
        self.root = root
        self.root.title("个人信息管理器")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # 设置样式
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("SimHei", 10))
        self.style.configure("TButton", font=("SimHei", 10))
        self.style.configure("TEntry", font=("SimHei", 10))
        self.style.configure("Header.TLabel", font=("SimHei", 16, "bold"))
        
        # 数据文件
        self.data_file = "personal_info.json"
        
        # 创建UI
        self.create_widgets()
        
        # 加载已有数据
        self.load_data()
    
    def create_widgets(self):
        # 标题
        header = ttk.Label(self.root, text="个人信息管理", style="Header.TLabel")
        header.pack(pady=10)
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 表单框架
        form_frame = ttk.LabelFrame(main_frame, text="个人信息", padding="10")
        form_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 姓名
        ttk.Label(form_frame, text="姓名:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.name_var, width=30).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # 年龄
        ttk.Label(form_frame, text="年龄:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.age_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.age_var, width=10).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # 性别
        ttk.Label(form_frame, text="性别:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.gender_var = tk.StringVar(value="男")
        gender_frame = ttk.Frame(form_frame)
        gender_frame.grid(row=2, column=1, sticky=tk.W, pady=5)
        ttk.Radiobutton(gender_frame, text="男", variable=self.gender_var, value="男").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(gender_frame, text="女", variable=self.gender_var, value="女").pack(side=tk.LEFT, padx=5)
        
        # 邮箱
        ttk.Label(form_frame, text="邮箱:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.email_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.email_var, width=30).grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(button_frame, text="保存信息", command=self.save_info).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="清除表单", command=self.clear_form).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="显示信息", command=self.show_info).pack(side=tk.LEFT, padx=5)
        
        # 信息显示区域
        display_frame = ttk.LabelFrame(main_frame, text="信息显示", padding="10")
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.info_text = tk.Text(display_frame, wrap=tk.WORD, width=50, height=10, font=("SimHei", 10))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        self.info_text.config(state=tk.DISABLED)
    
    def save_info(self):
        # 获取表单数据
        name = self.name_var.get().strip()
        age = self.age_var.get().strip()
        gender = self.gender_var.get()
        email = self.email_var.get().strip()
        
        # 验证数据
        if not name:
            messagebox.showerror("错误", "请输入姓名")
            return
        
        if age and not age.isdigit():
            messagebox.showerror("错误", "年龄必须是数字")
            return
        
        # 准备保存的数据
        info = {
            "name": name,
            "age": age if age else "未填写",
            "gender": gender,
            "email": email if email else "未填写"
        }
        
        # 保存到文件
        try:
            data = []
            if os.path.exists(self.data_file):
                with open(self.data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            
            data.append(info)
            
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo("成功", "信息已保存")
            self.clear_form()
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def clear_form(self):
        self.name_var.set("")
        self.age_var.set("")
        self.gender_var.set("男")
        self.email_var.set("")
    
    def show_info(self):
        try:
            if not os.path.exists(self.data_file):
                messagebox.showinfo("提示", "没有保存的信息")
                return
            
            with open(self.data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            
            if not data:
                self.info_text.insert(tk.END, "没有保存的信息")
            else:
                for i, info in enumerate(data, 1):
                    self.info_text.insert(tk.END, f"信息 {i}:\n")
                    self.info_text.insert(tk.END, f"姓名: {info['name']}\n")
                    self.info_text.insert(tk.END, f"年龄: {info['age']}\n")
                    self.info_text.insert(tk.END, f"性别: {info['gender']}\n")
                    self.info_text.insert(tk.END, f"邮箱: {info['email']}\n")
                    self.info_text.insert(tk.END, "-" * 40 + "\n")
            
            self.info_text.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("错误", f"读取失败: {str(e)}")
    
    def load_data(self):
        # 这里可以加载最后一次保存的数据到表单
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if data:
                    last_info = data[-1]
                    self.name_var.set(last_info["name"])
                    self.age_var.set(last_info["age"] if last_info["age"] != "未填写" else "")
                    self.gender_var.set(last_info["gender"])
                    self.email_var.set(last_info["email"] if last_info["email"] != "未填写" else "")
        except Exception as e:
            print(f"加载数据失败: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    # 确保中文显示正常
    app = PersonalInfoManager(root)
    root.mainloop()
