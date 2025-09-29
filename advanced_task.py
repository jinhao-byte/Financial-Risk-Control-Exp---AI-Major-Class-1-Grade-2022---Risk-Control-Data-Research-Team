import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os  # 新增：用于创建目录

# 1. 读取并合并数据
df_tr = pd.read_csv('data/train_transaction.csv')
df_id = pd.read_csv('data/train_identity.csv')
df = df_tr.merge(df_id, on='TransactionID', how='left')

# 2. 标签与特征分离
y = df['isFraud']
X = df.drop(['isFraud', 'TransactionID'], axis=1)

# 3. 缺失值处理
X = X.fillna(-999)

# 4. 类别特征编码（修改：保存编码器）
cat_encoders = {}  # 新增：存储类别编码器
cat_cols = X.select_dtypes('object').columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    cat_encoders[col] = le  # 新增：保存编码器到字典

# 5. 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# LightGBM

import lightgbm as lgb
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=64,
    colsample_bytree=0.7,
    subsample=0.7,
    random_state=42,
    class_weight='balanced'
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='F1-score',
    early_stopping_rounds=100,
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=100)
    ]
)
y_pred = model.predict(X_test)


class FraudTransformer(nn.Module):
    def __init__(self, num_numerical_features, categorical_dims, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.d_model = d_model

        # 输入层：数值特征投影 + 类别嵌入
        self.numerical_proj = nn.Linear(num_numerical_features, d_model)
        self.category_embs = nn.ModuleList([nn.Embedding(dim, d_model) for dim in categorical_dims])

        # 序列建模：位置编码 + Transformer编码层
        self.pos_encoding = nn.Parameter(torch.randn(5000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层：CLS token + 分类器
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, num_x, cat_x, mask=None):
        batch_size, seq_len = num_x.shape[0], num_x.shape[1]

        # 数值特征投影
        num_proj = self.numerical_proj(num_x)
        # 类别特征嵌入
        cat_embs = []
        for i, emb in enumerate(self.category_embs):
            cat_embs.append(emb(cat_x[:, :, i]))
        cat_proj = torch.stack(cat_embs).sum(dim=0)
        # 特征相加
        x = num_proj + cat_proj

        # 添加位置编码
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        # Transformer编码
        x = self.transformer(x, src_key_padding_mask=mask)


        # 扩展[CLS]标记
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # 添加到序列开头
        x = torch.cat([cls_tokens, x], dim=1)
        # 提取[CLS]标记输出
        cls_out = x[:, 0, :]
        # 交易欺诈预测
        return self.classifier(cls_out)

    # 初始化模型
    model = FraudTransformer(
        num_numerical_features=X_train_num[0].shape[1],
        categorical_dims=[100, 50, 20]
    )

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)

    # 加载最佳模型进行测试
    test_preds, test_labels = test_model(model, test_loader, device)
















# 7. 保存模型（修改：创建目录并保存）
if not os.path.exists('model'):
    os.makedirs('model')
joblib.dump(model, 'model/ieee_lgbm_model.pkl')

# 新增：保存类别编码器和特征列表
joblib.dump(cat_encoders, 'model/cat_encoders.pkl')  # 保存类别编码器
joblib.dump(X.columns.tolist(), 'model/feature_list.pkl')  # 保存特征列表
print("✅ 已保存预处理工具: cat_encoders.pkl 和 feature_list.pkl")

# 8. 模型评估
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("🔍 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("🎯 ROC AUC Score:", roc_auc_score(y_test, y_prob))

# 9. 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Fraud', 'Fraud'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.grid(False)
plt.show()

# 10. 精确率、召回率、F1-score 热力图展示
report = classification_report(y_test, y_pred, output_dict=True)
metrics_df = pd.DataFrame(report).transpose().loc[['0', '1'], ['precision', 'recall', 'f1-score']]

plt.figure(figsize=(6, 4))
sns.heatmap(metrics_df, annot=True, cmap='YlGnBu', fmt=".4f")
plt.title('Precision, Recall, F1-score by Class')
plt.ylabel('Class')
plt.xlabel('Metric')
plt.tight_layout()
plt.show()

# 11. 特征重要性图
lgb.plot_importance(model, max_num_features=20, importance_type='gain')
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()