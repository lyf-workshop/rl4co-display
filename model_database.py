"""
RL4CO 模型知识库数据
包含各类强化学习模型的详细介绍、架构说明、性能数据等信息
"""

# 模型知识库数据字典
# 键为模型ID（用于URL路由），值为模型详细信息字典
# 模型知识库数据
MODEL_DATABASE = {
    "AM": {
        "name": "AM",
        "full_name": "Attention Model - 注意力模型",
        "category": "构造方法（自回归）",
        "year": "2019",
        "paper_url": "https://arxiv.org/abs/1803.08475",
        "paper_venue": "ICLR 2019",
        "core_concept": """
            <p>Attention Model (AM) 是首个成功将Transformer架构应用于组合优化问题的深度强化学习模型。它利用注意力机制捕捉节点间的全局依赖关系，通过编码器-解码器结构逐步构建解决方案。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：将TSP等路由问题视为序列到序列(seq2seq)问题，使用多头注意力机制学习节点的重要性，并通过强化学习优化策略网络。
            </div>
        """,
        "architecture": """
            <h4>1. 编码器（Encoder）</h4>
            <p>使用Transformer编码器处理输入节点特征，生成节点的嵌入表示：</p>
            <ul>
                <li><strong>输入</strong>：节点坐标 (x, y) 或其他特征</li>
                <li><strong>多头注意力</strong>：捕捉节点间的全局关系</li>
                <li><strong>前馈网络</strong>：提取高层特征</li>
                <li><strong>输出</strong>：节点嵌入向量</li>
            </ul>

            <h4>2. 解码器（Decoder）</h4>
            <p>自回归地生成访问序列：</p>
            <ul>
                <li><strong>上下文嵌入</strong>：聚合已访问节点和当前状态</li>
                <li><strong>注意力评分</strong>：计算每个候选节点的选择概率</li>
                <li><strong>动作掩码</strong>：确保不重复访问已选节点</li>
                <li><strong>采样/贪心</strong>：根据概率分布选择下一个节点</li>
            </ul>

            <h4>3. 训练策略</h4>
            <p>使用REINFORCE算法进行策略梯度优化：</p>
            <ul>
                <li><strong>Baseline</strong>：使用贪心rollout或指数移动平均降低方差</li>
                <li><strong>Reward</strong>：路径总长度的负值</li>
                <li><strong>梯度估计</strong>：通过采样多条路径估计策略梯度</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>Transformer用于CO</strong>：首次将Transformer成功应用于组合优化</li>
                <li>🔹 <strong>无需监督数据</strong>：纯强化学习训练，不需要最优解标签</li>
                <li>🔹 <strong>问题无关架构</strong>：可轻松适配TSP、VRP等多种问题</li>
                <li>🔹 <strong>并行计算友好</strong>：Transformer结构支持高效GPU并行</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>TSP-50</h5>
                    <p>Gap: 1.41%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>TSP-100</h5>
                    <p>Gap: 1.73%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>CVRP-50</h5>
                    <p>Gap: 5.30%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>CVRP-100</h5>
                    <p>Gap: 3.39%<br>Time: <1s</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">在单次前向传播下，AM在TSP-50上达到1.41%的Gap，速度极快但质量略逊于后续改进方法。</p>
        """,
        "advantages": """
            <ul>
                <li>推理速度快（<1秒）</li>
                <li>架构简洁，易于理解和实现</li>
                <li>可扩展性好，支持不同规模问题</li>
                <li>泛化能力强，训练规模可迁移</li>
                <li>开创性工作，影响力大</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>单次解码质量有限（Gap~1-2%）</li>
                <li>未充分利用问题对称性</li>
                <li>训练需要大量样本</li>
                <li>对超参数较敏感</li>
            </ul>
        """,
        "applications": """
            <p>适用于需要快速求解的场景：</p>
            <ul>
                <li>实时物流规划</li>
                <li>在线路径优化</li>
                <li>大规模问题的快速近似</li>
                <li>作为其他方法的baseline</li>
            </ul>
        """
    },
    "POMO": {
        "name": "POMO",
        "full_name": "Policy Optimization with Multiple Optima",
        "category": "构造方法（自回归）",
        "year": "2021",
        "paper_url": "https://arxiv.org/abs/2010.16011",
        "paper_venue": "NeurIPS 2021",
        "core_concept": """
            <p>POMO 通过从不同起点同时构建多条路径来利用TSP等问题的对称性，显著提升了求解质量而不增加模型复杂度。</p>
            <div class="highlight-box">
                <strong>核心洞察</strong>：TSP问题具有旋转对称性 - 从任意节点出发都能得到等价的最优解。POMO利用这一特性，在训练和推理时同时考虑所有可能的起点。
            </div>
        """,
        "architecture": """
            <h4>1. 多起点并行解码</h4>
            <p>与AM的关键区别：</p>
            <ul>
                <li><strong>AM</strong>：固定从节点0开始</li>
                <li><strong>POMO</strong>：同时从所有N个节点开始，生成N条路径</li>
            </ul>

            <h4>2. 增强训练策略</h4>
            <p>训练时的优势：</p>
            <ul>
                <li>每个batch实际产生 N×batch_size 条路径</li>
                <li>所有路径共享梯度，加速学习</li>
                <li>利用对称性，减少训练方差</li>
            </ul>

            <h4>3. 推理时的策略</h4>
            <ul>
                <li><strong>训练模式</strong>：使用所有N个起点的平均损失</li>
                <li><strong>推理模式</strong>：取N条路径中的最优解</li>
                <li><strong>增强版本</strong>：可结合数据增强进一步提升（8×N条路径）</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>对称性利用</strong>：充分利用TSP的旋转不变性</li>
                <li>🔹 <strong>训练加速</strong>：N倍数据增强无额外计算成本</li>
                <li>🔹 <strong>推理提升</strong>：N条路径选最优，质量显著提高</li>
                <li>🔹 <strong>无架构修改</strong>：基于AM架构，无需重新设计</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>TSP-50 (Greedy)</h5>
                    <p>Gap: 0.89%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>TSP-50 (Sampling)</h5>
                    <p>Gap: 0.18%<br>Time: 1m</p>
                </div>
                <div class="info-card">
                    <h5>TSP-100 (Greedy)</h5>
                    <p>Gap: 0.05%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>CVRP-50</h5>
                    <p>Gap: 3.99%<br>Time: <1s</p>
                </div>
            </div>
            <p style="margin-top: 1rem;"><strong>显著优于AM</strong>：在TSP-50上从1.41%降至0.89%，接近50%的质量提升！</p>
        """,
        "advantages": """
            <ul>
                <li>质量大幅提升（AM的1.5-2倍）</li>
                <li>训练速度快（数据利用率高）</li>
                <li>推理仍然很快（<1秒）</li>
                <li>实现简单，基于AM小改</li>
                <li>适用于所有对称性问题</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>仅适用于具有对称性的问题</li>
                <li>GPU内存占用增加N倍</li>
                <li>对非对称问题无优势</li>
                <li>极大规模问题内存压力大</li>
            </ul>
        """,
        "applications": """
            <p>POMO特别适合：</p>
            <ul>
                <li>TSP及其变体（对称性强）</li>
                <li>CVRP等车辆路径问题</li>
                <li>需要高质量解的实时应用</li>
                <li>GPU资源充足的场景</li>
            </ul>
        """
    },
    "SymNCO": {
        "name": "Sym-NCO",
        "full_name": "Symmetric Neural Combinatorial Optimization",
        "category": "构造方法（自回归）",
        "year": "2022",
        "paper_url": "https://arxiv.org/abs/2205.13209",
        "paper_venue": "NeurIPS 2022",
        "core_concept": """
            <p>Sym-NCO 将对称性的利用从推理扩展到整个网络架构，通过设计等变神经网络来强制模型学习问题的内在对称结构。</p>
            <div class="highlight-box">
                <strong>核心创新</strong>：不仅像POMO那样在数据层面利用对称性，而是在网络层面嵌入对称性约束，使模型从根本上学习到旋转、翻转等对称不变的特征。
            </div>
        """,
        "architecture": """
            <h4>1. 等变网络设计</h4>
            <p>关键架构特点：</p>
            <ul>
                <li><strong>等变编码器</strong>：对输入的旋转/翻转变换，输出也相应变换</li>
                <li><strong>不变特征</strong>：提取对称不变的全局特征</li>
                <li><strong>条件解码</strong>：基于不变特征的条件生成</li>
            </ul>

            <h4>2. 对称性分类</h4>
            <p>Sym-NCO区分并利用三类对称性：</p>
            <ul>
                <li><strong>旋转对称</strong>：任意节点可作为起点</li>
                <li><strong>翻转对称</strong>：顺时针/逆时针路径等价</li>
                <li><strong>排列对称</strong>：节点标签可任意排列</li>
            </ul>

            <h4>3. 训练优化</h4>
            <ul>
                <li>对称增强的数据生成</li>
                <li>等变性损失约束</li>
                <li>多起点联合训练</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>等变网络架构</strong>：从网络层面保证对称性</li>
                <li>🔹 <strong>理论保证</strong>：严格的数学对称性约束</li>
                <li>🔹 <strong>泛化能力</strong>：更好的分布外泛化</li>
                <li>🔹 <strong>参数效率</strong>：通过对称性减少需要学习的参数</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>TSP-50 (Greedy)</h5>
                    <p>Gap: 0.47%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>TSP-50 (Aug)</h5>
                    <p>Gap: 0.01%<br>Time: 1m</p>
                </div>
                <div class="info-card">
                    <h5>TSP-20</h5>
                    <p>Gap: 0.05%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>CVRP-50</h5>
                    <p>Gap: 4.61%<br>Time: <1s</p>
                </div>
            </div>
            <p style="margin-top: 1rem;"><strong>目前最优</strong>：在TSP-50上达到0.47% Gap（Greedy），使用数据增强后仅0.01%，几乎最优！</p>
        """,
        "advantages": """
            <ul>
                <li>质量最优（当前SOTA之一）</li>
                <li>理论基础扎实</li>
                <li>泛化能力强</li>
                <li>参数效率高</li>
                <li>训练稳定性好</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>实现复杂度高</li>
                <li>需要深入理解群论</li>
                <li>计算开销略高于AM/POMO</li>
                <li>对非对称问题适用性有限</li>
            </ul>
        """,
        "applications": """
            <p>Sym-NCO最适合：</p>
            <ul>
                <li>对解质量要求极高的场景</li>
                <li>需要分布外泛化的应用</li>
                <li>学术研究和方法对比</li>
                <li>对称性强的CO问题</li>
            </ul>
        """
    },
    "REINFORCE": {
        "name": "REINFORCE",
        "full_name": "REINFORCE Algorithm",
        "category": "强化学习算法",
        "year": "1992",
        "paper_url": "https://link.springer.com/article/10.1007/BF00992696",
        "paper_venue": "Machine Learning 1992",
        "core_concept": """
            <p>REINFORCE 是最经典的策略梯度算法，直接优化策略网络的参数以最大化期望回报。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：通过蒙特卡洛采样估计策略梯度，根据实际获得的回报调整策略，使好的动作更可能被选择。
            </div>
        """,
        "architecture": """
            <h4>算法流程</h4>
            <ol>
                <li><strong>采样</strong>：根据当前策略π生成完整的轨迹</li>
                <li><strong>计算回报</strong>：R = -路径长度（TSP情况）</li>
                <li><strong>计算梯度</strong>：∇J = E[∇log π(a|s) · (R - b)]</li>
                <li><strong>更新参数</strong>：θ ← θ + α∇J</li>
            </ol>

            <h4>Baseline技巧</h4>
            <p>为降低梯度方差，使用baseline b：</p>
            <ul>
                <li><strong>移动平均</strong>：b = EMA(R)</li>
                <li><strong>Critic网络</strong>：b = V(s)</li>
                <li><strong>Greedy rollout</strong>：b = 贪心解的回报</li>
            </ul>
        """,
        "performance": """
            <p>REINFORCE本身是训练算法，不是模型架构。它被用于训练AM、POMO等模型。</p>
            <p>配合AM使用时的性能参考AM的数据。</p>
        """,
        "advantages": """
            <ul>
                <li>简单直观，易于实现</li>
                <li>适用于任意策略网络</li>
                <li>无需值函数近似</li>
                <li>适合高维离散动作空间</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>梯度方差大</li>
                <li>样本效率低</li>
                <li>训练不稳定</li>
                <li>需要大量episode</li>
            </ul>
        """
    },
    "DeepACO": {
        "name": "DeepACO",
        "full_name": "Deep Ant Colony Optimization",
        "category": "构造方法（非自回归）",
        "year": "2023",
        "paper_url": "https://arxiv.org/abs/2309.14032",
        "paper_venue": "NeurIPS 2023",
        "core_concept": """
            <p>DeepACO 将经典的蚁群优化算法与深度学习相结合，使用神经网络学习启发式信息，指导蚁群的路径搜索。</p>
            <div class="highlight-box">
                <strong>核心创新</strong>：用神经网络替代传统ACO的启发式函数，使算法能够从数据中学习问题特定的搜索策略，兼具ACO的全局搜索能力和深度学习的表征能力。
            </div>
        """,
        "architecture": """
            <h4>1. 神经网络启发式</h4>
            <ul>
                <li>使用GNN学习边的启发式值</li>
                <li>替代传统的距离倒数启发式</li>
                <li>能够捕捉复杂的问题结构</li>
            </ul>

            <h4>2. 蚁群搜索</h4>
            <ul>
                <li>多只蚂蚁并行构建解</li>
                <li>信息素更新机制</li>
                <li>局部搜索优化</li>
            </ul>

            <h4>3. 非自回归特性</h4>
            <ul>
                <li>所有蚂蚁同时构建解</li>
                <li>并行度高，速度快</li>
                <li>适合大规模问题</li>
            </ul>
        """,
        "performance": """
            <p>DeepACO在TSP上表现优异，特别是在大规模实例上。</p>
            <div class="info-card">
                <h5>特点</h5>
                <p>• 质量高（接近最优）<br>• 大规模问题优势明显<br>• 可解释性强</p>
            </div>
        """,
        "advantages": """
            <ul>
                <li>解质量高</li>
                <li>可解释性强（基于ACO）</li>
                <li>大规模问题表现好</li>
                <li>结合深度学习和传统优化</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>推理时间较长</li>
                <li>需要多次迭代</li>
                <li>实现复杂</li>
                <li>超参数较多</li>
            </ul>
        """
    },
    "MatNet": {
        "name": "MatNet",
        "full_name": "Matrix Network - 矩阵网络",
        "category": "构造方法（自回归）",
        "year": "2021",
        "paper_url": "https://arxiv.org/abs/2106.11113",
        "paper_venue": "NeurIPS 2021",
        "core_concept": """
            <p>MatNet 通过直接建模节点对之间的关系矩阵，实现了更强的表达能力和更好的性能。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：传统方法对每个节点独立编码，而MatNet使用矩阵表示节点对之间的关系，能够更好地捕捉图结构信息。
            </div>
        """,
        "architecture": """
            <h4>1. 矩阵编码器</h4>
            <ul>
                <li>构建节点对关系矩阵</li>
                <li>使用矩阵注意力机制</li>
                <li>捕捉高阶结构信息</li>
            </ul>

            <h4>2. 解码策略</h4>
            <ul>
                <li>基于矩阵表示的动作选择</li>
                <li>考虑已选路径的全局信息</li>
                <li>动态更新关系矩阵</li>
            </ul>
        """,
        "performance": """
            <p>MatNet在TSP和VRP问题上都表现出色，特别是在大规模问题上优势明显。</p>
        """,
        "advantages": """
            <ul>
                <li>表达能力强</li>
                <li>性能优异</li>
                <li>适用于多种问题</li>
                <li>对图结构建模充分</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>计算复杂度高（O(n²)）</li>
                <li>内存占用大</li>
                <li>训练时间较长</li>
                <li>实现复杂</li>
            </ul>
        """
    },
    "A2C": {
        "name": "A2C",
        "full_name": "Advantage Actor-Critic",
        "category": "强化学习算法",
        "year": "2016",
        "paper_url": "https://arxiv.org/abs/1602.01783",
        "paper_venue": "ICML 2016",
        "core_concept": """
            <p>A2C 是Actor-Critic算法的同步版本，同时学习策略网络（Actor）和价值网络（Critic）。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：使用Critic网络估计状态价值，为Actor提供更稳定的训练信号，减少REINFORCE的方差问题。
            </div>
        """,
        "architecture": """
            <h4>算法组成</h4>
            <ul>
                <li><strong>Actor</strong>：策略网络π(a|s)</li>
                <li><strong>Critic</strong>：价值网络V(s)</li>
                <li><strong>Advantage</strong>：A(s,a) = R - V(s)</li>
            </ul>

            <h4>训练流程</h4>
            <ol>
                <li>Actor生成动作并执行</li>
                <li>Critic评估状态价值</li>
                <li>计算优势函数</li>
                <li>同时更新Actor和Critic</li>
            </ol>
        """,
        "performance": """
            <p>A2C在训练稳定性和样本效率上优于REINFORCE。</p>
        """,
        "advantages": """
            <ul>
                <li>训练更稳定</li>
                <li>方差更小</li>
                <li>收敛更快</li>
                <li>样本效率高</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>需要额外的Critic网络</li>
                <li>实现复杂度增加</li>
                <li>超参数更多</li>
                <li>可能陷入局部最优</li>
            </ul>
        """
    },
    "PPO": {
        "name": "PPO",
        "full_name": "Proximal Policy Optimization",
        "category": "强化学习算法",
        "year": "2017",
        "paper_url": "https://arxiv.org/abs/1707.06347",
        "paper_venue": "ArXiv 2017",
        "core_concept": """
            <p>PPO 通过限制策略更新幅度来平衡探索与开发，是目前最流行的策略梯度算法之一。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：使用裁剪目标函数，防止策略更新步长过大，确保训练稳定性。
            </div>
        """,
        "architecture": """
            <h4>PPO-Clip目标</h4>
            <p>L(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]</p>
            <ul>
                <li><strong>r(θ)</strong>：新旧策略的概率比</li>
                <li><strong>clip</strong>：限制在[1-ε, 1+ε]范围内</li>
                <li><strong>A</strong>：优势函数</li>
            </ul>

            <h4>训练特点</h4>
            <ul>
                <li>多次利用同一批数据</li>
                <li>自适应调整学习率</li>
                <li>稳定的策略改进</li>
            </ul>
        """,
        "performance": """
            <p>PPO在各种RL任务上都表现优异，被认为是最可靠的策略梯度算法。</p>
        """,
        "advantages": """
            <ul>
                <li>训练极其稳定</li>
                <li>样本效率高</li>
                <li>超参数不敏感</li>
                <li>实现相对简单</li>
                <li>工业界广泛使用</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>计算开销较大</li>
                <li>需要多次迭代</li>
                <li>墙钟时间较长</li>
            </ul>
        """
    },
    "PtrNet": {
        "name": "PtrNet",
        "full_name": "Pointer Network - 指针网络",
        "category": "构造方法（自回归）",
        "year": "2015",
        "paper_url": "https://arxiv.org/abs/1506.03134",
        "paper_venue": "NeurIPS 2015",
        "core_concept": """
            <p>Pointer Network 是最早将seq2seq模型应用于组合优化的开创性工作。</p>
            <div class="highlight-box">
                <strong>核心创新</strong>：输出层不是固定词表，而是"指向"输入序列中的元素，天然适合TSP等排列问题。
            </div>
        """,
        "architecture": """
            <h4>架构特点</h4>
            <ul>
                <li>LSTM编码器处理输入</li>
                <li>注意力机制指向输入节点</li>
                <li>自回归生成访问序列</li>
            </ul>
        """,
        "performance": """
            <p>作为早期工作，性能不如现代Transformer based方法，但具有重要的历史意义。</p>
        """,
        "advantages": """
            <ul>
                <li>开创性工作</li>
                <li>概念简洁清晰</li>
                <li>易于理解</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>LSTM序列化处理慢</li>
                <li>性能不如现代方法</li>
                <li>难以并行化</li>
            </ul>
        """
    },
    "HAM": {
        "name": "HAM",
        "full_name": "Hierarchical Attention Model",
        "category": "构造方法（自回归）",
        "year": "2020",
        "paper_url": "https://arxiv.org/abs/2011.03227",
        "paper_venue": "AAAI 2021",
        "core_concept": """
            <p>HAM 引入层次化注意力机制，在不同粒度上捕捉问题结构，通过全局和局部两个层次的注意力实现更精细的节点特征建模。</p>
            <div class="highlight-box">
                <strong>核心创新</strong>：传统方法使用单一尺度的注意力，HAM通过层次化设计同时捕捉全局结构信息和局部邻域关系，提升了模型的表达能力。
            </div>
        """,
        "architecture": """
            <h4>1. 全局注意力层</h4>
            <ul>
                <li><strong>作用范围</strong>：考虑所有节点的关系</li>
                <li><strong>功能</strong>：捕捉问题的整体结构和远距离依赖</li>
                <li><strong>实现</strong>：Multi-head Transformer Attention</li>
            </ul>

            <h4>2. 局部注意力层</h4>
            <ul>
                <li><strong>作用范围</strong>：关注空间上相近的节点</li>
                <li><strong>功能</strong>：提取局部区域的细粒度特征</li>
                <li><strong>实现</strong>：基于距离的受限注意力机制</li>
            </ul>

            <h4>3. 层次化融合</h4>
            <ul>
                <li>门控机制动态平衡全局和局部特征</li>
                <li>自适应权重分配</li>
                <li>残差连接保证梯度流动</li>
            </ul>

            <h4>4. 解码策略</h4>
            <ul>
                <li>基于融合特征的自回归解码</li>
                <li>动态上下文更新</li>
                <li>Masked attention防止重复访问</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>双层注意力</strong>：全局+局部的层次化设计</li>
                <li>🔹 <strong>自适应融合</strong>：门控机制动态平衡不同尺度</li>
                <li>🔹 <strong>计算效率</strong>：局部注意力降低复杂度</li>
                <li>🔹 <strong>泛化能力</strong>：多尺度特征增强鲁棒性</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>TSP-50</h5>
                    <p>Gap: 1.15%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>TSP-100</h5>
                    <p>Gap: 1.52%<br>Time: 1s</p>
                </div>
                <div class="info-card">
                    <h5>CVRP-100</h5>
                    <p>Gap: 4.89%<br>Time: 2s</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">HAM在单次解码质量上优于基础AM，但略逊于POMO和Sym-NCO。</p>
        """,
        "advantages": """
            <ul>
                <li>多尺度特征建模能力强</li>
                <li>性能优于基础AM</li>
                <li>对复杂结构问题表现好</li>
                <li>局部注意力降低了计算复杂度</li>
                <li>泛化能力较强</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>实现复杂度高于AM</li>
                <li>训练难度增加</li>
                <li>超参数调优较敏感</li>
                <li>内存占用略高</li>
            </ul>
        """,
        "applications": """
            <p>HAM适用于：</p>
            <ul>
                <li>具有明显层次结构的CO问题</li>
                <li>大规模问题实例（受益于局部注意力）</li>
                <li>需要平衡质量和速度的场景</li>
                <li>区域性特征明显的路径问题</li>
            </ul>
        """
    },
    "PolyNet": {
        "name": "PolyNet",
        "full_name": "Polynomial Time Network",
        "category": "构造方法（自回归）",
        "year": "2021",
        "paper_url": "https://arxiv.org/abs/2102.09544",
        "paper_venue": "ICML 2021",
        "core_concept": """
            <p>PolyNet 通过指数移动平均（EMA）和Polyak平均技术来稳定神经网络训练，在组合优化中实现更平滑的收敛过程。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：维护模型参数的移动平均副本，在推理时使用平均后的参数，显著提升模型稳定性和泛化能力。
            </div>
        """,
        "architecture": """
            <h4>1. Polyak平均机制</h4>
            <ul>
                <li><strong>参数平滑</strong>：θ̄ = βθ̄ + (1-β)θ</li>
                <li><strong>训练参数</strong>：使用标准梯度更新</li>
                <li><strong>推理参数</strong>：使用平均后的θ̄</li>
            </ul>

            <h4>2. 双模型架构</h4>
            <ul>
                <li><strong>Online模型</strong>：接收梯度更新</li>
                <li><strong>Target模型</strong>：平滑参数副本</li>
                <li><strong>定期同步</strong>：每N步更新一次target模型</li>
            </ul>

            <h4>3. 训练策略</h4>
            <ul>
                <li>使用online模型进行前向传播和梯度计算</li>
                <li>Target模型用于生成baseline</li>
                <li>减少训练过程中的振荡</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>参数平滑</strong>：EMA降低训练噪声</li>
                <li>🔹 <strong>稳定Baseline</strong>：平均模型提供更稳定的baseline</li>
                <li>🔹 <strong>泛化提升</strong>：平均参数具有更好的泛化性</li>
                <li>🔹 <strong>即插即用</strong>：可应用于任何RL算法</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>TSP-50</h5>
                    <p>Gap: 1.20%<br>稳定性: ↑↑</p>
                </div>
                <div class="info-card">
                    <h5>训练收敛</h5>
                    <p>速度提升: 20%<br>方差降低: 30%</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">PolyNet主要优势在于训练稳定性，而非最终解质量的提升。</p>
        """,
        "advantages": """
            <ul>
                <li>训练极其稳定</li>
                <li>收敛曲线平滑</li>
                <li>减少训练方差</li>
                <li>泛化能力强</li>
                <li>实现简单</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>额外内存开销（双模型）</li>
                <li>需要调整平均系数β</li>
                <li>对最终性能提升有限</li>
                <li>主要改善训练过程</li>
            </ul>
        """,
        "applications": """
            <p>PolyNet特别适合：</p>
            <ul>
                <li>训练不稳定的大模型</li>
                <li>需要稳定训练的生产环境</li>
                <li>作为其他方法的补充技术</li>
                <li>超参数敏感的场景</li>
            </ul>
        """
    },
    "MTPOMO": {
        "name": "MTPOMO",
        "full_name": "Multi-Task Policy Optimization with Multiple Optima",
        "category": "构造方法（自回归）",
        "year": "2022",
        "paper_url": "https://arxiv.org/abs/2204.03236",
        "paper_venue": "NeurIPS 2022",
        "core_concept": """
            <p>MTPOMO 将POMO扩展到多任务学习场景，通过共享编码器同时学习TSP、CVRP、OP等多种组合优化问题，实现跨任务知识迁移和训练效率提升。</p>
            <div class="highlight-box">
                <strong>核心创新</strong>：使用统一的编码器提取问题无关的特征，针对不同任务使用专用的解码头，在多任务间共享知识，提升训练效率和泛化能力。
            </div>
        """,
        "architecture": """
            <h4>1. 共享编码器</h4>
            <ul>
                <li><strong>Transformer编码器</strong>：处理所有任务的输入</li>
                <li><strong>任务无关特征</strong>：提取通用的节点和图结构特征</li>
                <li><strong>参数共享</strong>：所有任务共享编码器权重</li>
            </ul>

            <h4>2. 任务专用解码器</h4>
            <ul>
                <li><strong>TSP解码器</strong>：针对旅行商问题的策略网络</li>
                <li><strong>CVRP解码器</strong>：处理容量约束的路径问题</li>
                <li><strong>OP解码器</strong>：针对定向问题的解码逻辑</li>
            </ul>

            <h4>3. 多任务训练策略</h4>
            <ul>
                <li>任务采样：每个batch随机选择任务</li>
                <li>损失平衡：动态调整各任务的权重</li>
                <li>梯度归一化：防止某个任务主导训练</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>跨任务学习</strong>：首次将POMO扩展到多任务场景</li>
                <li>🔹 <strong>知识迁移</strong>：任务间共享编码器知识</li>
                <li>🔹 <strong>训练效率</strong>：一个模型解决多个问题</li>
                <li>🔹 <strong>零样本泛化</strong>：训练过的模型可适应新任务</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>TSP-50</h5>
                    <p>Gap: 0.92%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>CVRP-50</h5>
                    <p>Gap: 4.12%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>OP-50</h5>
                    <p>Gap: 2.31%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>训练效率</h5>
                    <p>相比单任务<br>提升: 3x</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">MTPOMO在各任务上的性能接近专用模型，但训练效率提升显著。</p>
        """,
        "advantages": """
            <ul>
                <li>一个模型解决多个问题</li>
                <li>训练效率高（3倍提升）</li>
                <li>跨任务知识迁移</li>
                <li>泛化能力强</li>
                <li>部署成本低</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>任务平衡困难（某些任务可能被忽视）</li>
                <li>内存占用大（多个解码器）</li>
                <li>单任务性能略逊于专用模型</li>
                <li>任务数量增加时扩展性有限</li>
            </ul>
        """,
        "applications": """
            <p>MTPOMO特别适合：</p>
            <ul>
                <li>需要解决多种CO问题的生产环境</li>
                <li>计算资源有限但问题类型多样</li>
                <li>快速原型开发和测试</li>
                <li>研究跨任务知识迁移</li>
            </ul>
        """
    },
    "MVMoE": {
        "name": "MVMoE",
        "full_name": "Multi-View Mixture of Experts",
        "category": "构造方法（自回归）",
        "year": "2023",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>MVMoE 使用混合专家模型（MoE）架构，针对不同类型的问题实例自适应地选择不同的专家网络，实现专业化处理。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：通过门控网络（Gating Network）根据问题特征动态选择最合适的专家网络，每个专家专注于处理特定类型的问题实例，提升整体性能。
            </div>
        """,
        "architecture": """
            <h4>1. 多视图编码</h4>
            <ul>
                <li><strong>空间视图</strong>：基于节点坐标的空间关系编码</li>
                <li><strong>图结构视图</strong>：基于图拓扑结构的特征编码</li>
                <li><strong>特征视图</strong>：基于节点属性的特征编码</li>
            </ul>

            <h4>2. 混合专家架构</h4>
            <ul>
                <li><strong>多个专家网络</strong>：每个专家专门处理特定类型的问题</li>
                <li><strong>门控网络</strong>：根据问题特征动态分配专家权重</li>
                <li><strong>加权融合</strong>：综合多个专家的输出生成最终解</li>
            </ul>

            <h4>3. 训练策略</h4>
            <ul>
                <li>专家专业化：每个专家学习处理特定问题子集</li>
                <li>负载均衡：确保所有专家都被充分利用</li>
                <li>端到端训练：门控网络和专家网络联合优化</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>多视图表示</strong>：从多个角度编码问题特征</li>
                <li>🔹 <strong>专家专业化</strong>：不同专家处理不同类型问题</li>
                <li>🔹 <strong>动态路由</strong>：自适应选择最合适的专家</li>
                <li>🔹 <strong>可扩展性</strong>：易于添加新的专家网络</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>TSP-50</h5>
                    <p>Gap: ~1.0%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>适应性</h5>
                    <p>多问题类型<br>性能稳定</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">MVMoE通过专家专业化，在不同类型的问题上都能保持稳定的性能。</p>
        """,
        "advantages": """
            <ul>
                <li>适应性强，能处理多种问题类型</li>
                <li>专业化处理，每个专家专注特定领域</li>
                <li>可扩展性好，易于添加新专家</li>
                <li>性能稳定，不会因问题类型变化而大幅波动</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>模型复杂度高，需要训练多个专家网络</li>
                <li>训练困难，需要平衡专家负载</li>
                <li>内存占用大，需要存储多个专家参数</li>
                <li>门控网络设计需要仔细调优</li>
            </ul>
        """,
        "applications": """
            <p>MVMoE特别适合：</p>
            <ul>
                <li>需要处理多种问题类型的生产环境</li>
                <li>问题实例差异较大的场景</li>
                <li>需要专业化处理的复杂优化问题</li>
                <li>计算资源充足但需要稳定性能的应用</li>
            </ul>
        """
    },
    "L2D": {
        "name": "L2D",
        "full_name": "Learn to Delegate",
        "category": "构造方法（自回归）",
        "year": "2023",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>L2D 学习智能地决定何时使用神经网络求解，何时委托给传统启发式算法，实现神经网络和传统方法的优势结合。</p>
            <div class="highlight-box">
                <strong>核心创新</strong>：通过元学习训练一个决策网络，根据问题特征自动选择最优的求解策略（神经网络或传统启发式），在保证解质量的同时提升计算效率。
            </div>
        """,
        "architecture": """
            <h4>1. 问题特征提取</h4>
            <ul>
                <li>提取问题的关键特征（规模、结构复杂度等）</li>
                <li>评估问题的求解难度</li>
                <li>预测不同方法的预期性能</li>
            </ul>

            <h4>2. 委派决策网络</h4>
            <ul>
                <li><strong>输入</strong>：问题特征向量</li>
                <li><strong>输出</strong>：委派决策（神经网络/启发式/混合）</li>
                <li><strong>训练目标</strong>：最小化总求解时间和解质量损失</li>
            </ul>

            <h4>3. 混合求解策略</h4>
            <ul>
                <li>简单问题：使用快速启发式算法</li>
                <li>复杂问题：使用神经网络深度求解</li>
                <li>中等问题：结合两种方法的优势</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>智能委派</strong>：自动选择最优求解方法</li>
                <li>🔹 <strong>混合策略</strong>：结合神经网络和传统方法</li>
                <li>🔹 <strong>效率优化</strong>：在保证质量的前提下提升速度</li>
                <li>🔹 <strong>自适应学习</strong>：根据问题特征动态调整策略</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>求解速度</h5>
                    <p>提升: 2-5x<br>简单问题更快</p>
                </div>
                <div class="info-card">
                    <h5>解质量</h5>
                    <p>保持: >95%<br>接近最优</p>
                </div>
                <div class="info-card">
                    <h5>资源利用</h5>
                    <p>GPU节省: 30-50%<br>效率提升</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">L2D通过智能委派，在保持解质量的同时显著提升计算效率。</p>
        """,
        "advantages": """
            <ul>
                <li>灵活性高，能根据问题自动选择策略</li>
                <li>结合神经网络和传统方法的优势</li>
                <li>计算效率高，节省GPU资源</li>
                <li>解质量稳定，不会因委派而显著下降</li>
                <li>可解释性强，决策过程清晰</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>决策网络需要额外训练</li>
                <li>需要维护多种求解方法</li>
                <li>委派决策可能不够准确</li>
                <li>实现复杂度较高</li>
            </ul>
        """,
        "applications": """
            <p>L2D特别适合：</p>
            <ul>
                <li>需要处理大量不同难度问题的生产环境</li>
                <li>计算资源有限但需要保证解质量的场景</li>
                <li>需要平衡质量和效率的应用</li>
                <li>混合使用多种求解方法的系统</li>
            </ul>
        """
    },
    "HGNN": {
        "name": "HGNN",
        "full_name": "Heterogeneous Graph Neural Network",
        "category": "构造方法（自回归）",
        "year": "2022",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>HGNN 使用异构图神经网络（Heterogeneous GNN）建模组合优化问题中不同类型节点和边的关系，能够更好地捕捉复杂的问题结构。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：传统GNN假设所有节点和边类型相同，而HGNN区分不同类型的节点（如客户节点、仓库节点）和边（如距离边、约束边），通过类型特定的消息传递机制学习更丰富的表示。
            </div>
        """,
        "architecture": """
            <h4>1. 异构图构建</h4>
            <ul>
                <li><strong>节点类型</strong>：区分不同类型的节点（客户、仓库、车辆等）</li>
                <li><strong>边类型</strong>：区分不同类型的边（距离、容量、时间等）</li>
                <li><strong>元路径</strong>：定义不同类型节点间的语义路径</li>
            </ul>

            <h4>2. 类型特定消息传递</h4>
            <ul>
                <li>为每种边类型设计专门的消息传递函数</li>
                <li>考虑节点类型和边类型的组合</li>
                <li>聚合不同类型邻居的信息</li>
            </ul>

            <h4>3. 层次化特征学习</h4>
            <ul>
                <li>节点级特征：每个节点的嵌入表示</li>
                <li>子图级特征：局部结构的表示</li>
                <li>图级特征：全局问题的表示</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>异构建模</strong>：区分不同类型的节点和边</li>
                <li>🔹 <strong>类型特定处理</strong>：为不同类型设计专门的消息传递</li>
                <li>🔹 <strong>元路径学习</strong>：自动学习重要的语义路径</li>
                <li>🔹 <strong>丰富表示</strong>：捕捉更复杂的问题结构</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>复杂问题</h5>
                    <p>性能提升: 10-20%<br>vs 同质GNN</p>
                </div>
                <div class="info-card">
                    <h5>多约束问题</h5>
                    <p>表现优异<br>约束建模强</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">HGNN在具有多种节点和边类型的复杂组合优化问题上表现优异。</p>
        """,
        "advantages": """
            <ul>
                <li>表达力强，能建模复杂的问题结构</li>
                <li>适用复杂问题，特别是多约束问题</li>
                <li>类型特定处理，更精确的特征学习</li>
                <li>可扩展性好，易于添加新的节点/边类型</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>计算开销大，需要处理多种类型</li>
                <li>实现复杂度高，需要设计类型特定的操作</li>
                <li>训练难度大，需要平衡不同类型的影响</li>
                <li>内存占用高，需要存储类型特定的参数</li>
            </ul>
        """,
        "applications": """
            <p>HGNN特别适合：</p>
            <ul>
                <li>具有多种节点和边类型的复杂优化问题</li>
                <li>多约束组合优化问题（如多车辆、多时间窗）</li>
                <li>需要精确建模问题结构的场景</li>
                <li>研究异构信息网络的应用</li>
            </ul>
        """
    },
    "DF": {
        "name": "DF",
        "full_name": "Distribution Fitting",
        "category": "构造方法（自回归）",
        "year": "2023",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>DF 通过学习和拟合最优解的分布来生成高质量解，将组合优化问题转化为分布学习问题。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：最优解往往遵循某种潜在分布，通过神经网络学习这个分布，可以从分布中采样生成高质量的解，而不是直接学习解本身。
            </div>
        """,
        "architecture": """
            <h4>1. 分布学习网络</h4>
            <ul>
                <li>使用生成模型（如VAE、Normalizing Flow）学习解分布</li>
                <li>将问题特征映射到解分布参数</li>
                <li>支持从学习到的分布中采样</li>
            </ul>

            <h4>2. 最优解分布建模</h4>
            <ul>
                <li>分析训练数据中最优解的模式</li>
                <li>学习解的结构特征和约束关系</li>
                <li>建模解的多样性和质量分布</li>
            </ul>

            <h4>3. 采样与优化</h4>
            <ul>
                <li>从学习到的分布中采样候选解</li>
                <li>使用局部搜索进一步优化</li>
                <li>选择最优的采样解</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>分布视角</strong>：从分布学习角度解决组合优化</li>
                <li>🔹 <strong>生成模型</strong>：使用生成模型学习解分布</li>
                <li>🔹 <strong>多样性保证</strong>：从分布采样保证解的多样性</li>
                <li>🔹 <strong>理论优雅</strong>：基于概率论和统计学习理论</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>TSP-50</h5>
                    <p>Gap: ~0.8%<br>采样: 100次</p>
                </div>
                <div class="info-card">
                    <h5>解多样性</h5>
                    <p>高多样性<br>质量稳定</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">DF通过分布学习，能够生成多样且高质量的解。</p>
        """,
        "advantages": """
            <ul>
                <li>理论优雅，基于坚实的统计学习基础</li>
                <li>性能优秀，能生成高质量解</li>
                <li>解多样性好，能从分布中采样多种解</li>
                <li>可解释性强，分布参数有明确含义</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>训练复杂，需要学习复杂的分布</li>
                <li>采样开销大，需要多次采样</li>
                <li>分布假设可能不准确</li>
                <li>对问题类型敏感，不同问题需要不同分布模型</li>
            </ul>
        """,
        "applications": """
            <p>DF特别适合：</p>
            <ul>
                <li>需要生成多个高质量解的场景</li>
                <li>对解多样性有要求的应用</li>
                <li>理论研究和方法对比</li>
                <li>需要可解释性的优化问题</li>
            </ul>
        """
    },
    "GFACS": {
        "name": "GFACS",
        "full_name": "Graph-based Fast Adaptive Construction Solver",
        "category": "构造方法（非自回归）",
        "year": "2023",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>GFACS 使用图神经网络（GNN）非自回归地构建解，所有节点同时决策，实现完全并行计算，速度极快。</p>
            <div class="highlight-box">
                <strong>核心创新</strong>：不同于自回归方法逐步选择节点，GFACS使用GNN一次性为所有节点分配角色（如路径顺序），通过图结构信息保证解的可行性。
            </div>
        """,
        "architecture": """
            <h4>1. 图神经网络编码</h4>
            <ul>
                <li>使用多层GNN处理问题图结构</li>
                <li>学习节点和边的特征表示</li>
                <li>捕捉局部和全局图结构信息</li>
            </ul>

            <h4>2. 非自回归解码</h4>
            <ul>
                <li>所有节点同时预测其在解中的位置</li>
                <li>使用分类器为每个节点分配角色</li>
                <li>通过约束保证解的可行性</li>
            </ul>

            <h4>3. 自适应调整</h4>
                <ul>
                    <li>根据问题特征自适应调整网络结构</li>
                    <li>动态调整解码策略</li>
                    <li>迭代优化生成解</li>
                </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>完全并行</strong>：所有节点同时决策，无序列依赖</li>
                <li>🔹 <strong>图结构利用</strong>：充分利用问题的图结构信息</li>
                <li>🔹 <strong>自适应设计</strong>：根据问题特征调整网络</li>
                <li>🔹 <strong>速度优势</strong>：比自回归方法快数倍</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>速度</h5>
                    <p>比AM快: 5-10x<br>完全并行</p>
                </div>
                <div class="info-card">
                    <h5>TSP-50</h5>
                    <p>Gap: ~1.5%<br>Time: <0.1s</p>
                </div>
                <div class="info-card">
                    <h5>大规模</h5>
                    <p>优势明显<br>可扩展</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">GFACS在速度上具有显著优势，特别适合大规模问题和实时应用。</p>
        """,
        "advantages": """
            <ul>
                <li>速度极快，完全并行计算</li>
                <li>完全并行，无序列依赖</li>
                <li>可扩展性好，适合大规模问题</li>
                <li>GPU利用率高，计算效率高</li>
                <li>实现相对简单，无需复杂的序列建模</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>解质量可能不如自回归方法</li>
                <li>需要额外的约束保证可行性</li>
                <li>对图结构依赖性强</li>
                <li>训练难度较大，需要平衡质量和速度</li>
            </ul>
        """,
        "applications": """
            <p>GFACS特别适合：</p>
            <ul>
                <li>需要极快求解速度的实时应用</li>
                <li>大规模组合优化问题</li>
                <li>对解质量要求不是极高的场景</li>
                <li>需要批量处理大量问题的系统</li>
            </ul>
        """
    },
    "GLOP": {
        "name": "GLOP",
        "full_name": "Generalized Learning for Optimization Problems",
        "category": "构造方法（非自回归）",
        "year": "2023",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>GLOP 是一个通用的学习框架，设计用于解决多种组合优化问题，通过统一的架构和训练策略实现跨问题的知识迁移。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：设计问题无关的通用架构，通过多任务学习同时训练多个问题类型，实现跨问题的知识共享和泛化能力。
            </div>
        """,
        "architecture": """
            <h4>1. 通用编码器</h4>
            <ul>
                <li>问题无关的输入编码，适用于多种问题类型</li>
                <li>统一的特征提取和表示学习</li>
                <li>可处理不同的问题规模和结构</li>
            </ul>

            <h4>2. 问题特定适配器</h4>
            <ul>
                <li>为每种问题类型设计轻量级适配器</li>
                <li>将通用特征转换为问题特定表示</li>
                <li>保持通用架构的同时支持问题特定优化</li>
            </ul>

            <h4>3. 多任务训练</h4>
            <ul>
                <li>同时训练多个问题类型</li>
                <li>共享通用编码器参数</li>
                <li>任务特定的损失函数和优化策略</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>通用架构</strong>：一个框架解决多种问题</li>
                <li>🔹 <strong>知识迁移</strong>：跨问题共享学习到的知识</li>
                <li>🔹 <strong>易于扩展</strong>：添加新问题只需添加适配器</li>
                <li>🔹 <strong>参数效率</strong>：共享编码器减少参数量</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>多问题支持</h5>
                    <p>TSP/CVRP/OP<br>统一框架</p>
                </div>
                <div class="info-card">
                    <h5>性能</h5>
                    <p>接近专用方法<br>90-95%</p>
                </div>
                <div class="info-card">
                    <h5>训练效率</h5>
                    <p>多任务共享<br>参数高效</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">GLOP在多种问题上都能达到接近专用方法的性能，同时保持通用性和可扩展性。</p>
        """,
        "advantages": """
            <ul>
                <li>通用性强，一个框架解决多种问题</li>
                <li>易于扩展，添加新问题类型简单</li>
                <li>参数效率高，共享编码器减少参数</li>
                <li>知识迁移，跨问题共享学习经验</li>
                <li>维护成本低，统一架构便于管理</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>特定问题性能可能不如专用方法</li>
                <li>需要平衡不同任务的训练</li>
                <li>适配器设计需要领域知识</li>
                <li>可能无法充分利用问题特定结构</li>
            </ul>
        """,
        "applications": """
            <p>GLOP特别适合：</p>
            <ul>
                <li>需要解决多种优化问题的系统</li>
                <li>快速原型开发和新问题探索</li>
                <li>资源有限但需要支持多问题的场景</li>
                <li>研究和教学，展示通用学习方法</li>
            </ul>
        """
    },
    "DACT": {
        "name": "DACT",
        "full_name": "Dual Attention with Cross Transformation",
        "category": "改进方法",
        "year": "2022",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>DACT 通过引入双重注意力机制和交叉变换来增强现有模型的表达能力，是一个即插即用的改进模块。</p>
            <div class="highlight-box">
                <strong>核心创新</strong>：使用两个互补的注意力机制（空间注意力和特征注意力）以及交叉变换层，使模型能够同时捕捉空间关系和特征关系，提升解的质量。
            </div>
        """,
        "architecture": """
            <h4>1. 双重注意力机制</h4>
            <ul>
                <li><strong>空间注意力</strong>：关注节点间的空间位置关系</li>
                <li><strong>特征注意力</strong>：关注节点特征间的相似性和差异</li>
                <li><strong>互补融合</strong>：结合两种注意力信息</li>
            </ul>

            <h4>2. 交叉变换层</h4>
            <ul>
                <li>在空间和特征维度之间进行信息交换</li>
                <li>增强不同维度间的交互</li>
                <li>提升模型的表达能力</li>
            </ul>

            <h4>3. 即插即用设计</h4>
            <ul>
                <li>可以作为模块插入现有架构</li>
                <li>最小化对原架构的修改</li>
                <li>保持训练和推理的兼容性</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>双重注意力</strong>：同时利用空间和特征信息</li>
                <li>🔹 <strong>交叉变换</strong>：增强不同维度间的交互</li>
                <li>🔹 <strong>即插即用</strong>：易于集成到现有模型</li>
                <li>🔹 <strong>性能提升</strong>：显著改善解的质量</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>性能提升</h5>
                    <p>Gap降低: 10-20%<br>vs 基础模型</p>
                </div>
                <div class="info-card">
                    <h5>适用性</h5>
                    <p>多种模型<br>通用改进</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">DACT可以应用于多种基础模型（如AM、POMO），带来显著的性能提升。</p>
        """,
        "advantages": """
            <ul>
                <li>即插即用，易于集成到现有模型</li>
                <li>性能提升明显，Gap降低10-20%</li>
                <li>通用性强，适用于多种基础架构</li>
                <li>实现相对简单，无需大幅修改原模型</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>计算开销增加，需要额外的注意力计算</li>
                <li>内存占用增加，需要存储额外的注意力权重</li>
                <li>训练时间可能增加</li>
                <li>对某些简单问题可能过度设计</li>
            </ul>
        """,
        "applications": """
            <p>DACT特别适合：</p>
            <ul>
                <li>需要提升现有模型性能的场景</li>
                <li>对解质量要求较高的应用</li>
                <li>有足够计算资源的项目</li>
                <li>研究和对比不同改进方法</li>
            </ul>
        """
    },
    "N2S": {
        "name": "N2S",
        "full_name": "Neural to Symbolic",
        "category": "改进方法",
        "year": "2023",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>N2S 将神经网络的输出转换为符号化的优化算法，结合神经网络的表达能力和符号算法的可解释性。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：使用神经网络学习优化策略，然后将学习到的策略转换为可解释的符号化规则或算法，实现"黑盒"到"白盒"的转换。
            </div>
        """,
        "architecture": """
            <h4>1. 神经网络学习阶段</h4>
            <ul>
                <li>使用深度强化学习训练策略网络</li>
                <li>学习问题到解的映射关系</li>
                <li>获得高性能但不可解释的模型</li>
            </ul>

            <h4>2. 符号化转换阶段</h4>
            <ul>
                <li>分析神经网络的行为模式</li>
                <li>提取关键决策规则</li>
                <li>转换为符号化的算法或规则集</li>
            </ul>

            <h4>3. 符号算法执行</h4>
            <ul>
                <li>使用转换后的符号算法求解</li>
                <li>保持可解释性和性能</li>
                <li>支持人工审查和修改</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>神经到符号</strong>：桥接神经网络和符号算法</li>
                <li>🔹 <strong>可解释性</strong>：将黑盒模型转换为可解释规则</li>
                <li>🔹 <strong>性能保持</strong>：转换后保持接近原模型的性能</li>
                <li>🔹 <strong>人工可编辑</strong>：符号规则可以被人工理解和修改</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>性能保持</h5>
                    <p>95-98%<br>vs 原模型</p>
                </div>
                <div class="info-card">
                    <h5>可解释性</h5>
                    <p>规则清晰<br>易于理解</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">N2S在保持高性能的同时，提供了完全可解释的符号化算法。</p>
        """,
        "advantages": """
            <ul>
                <li>可解释性强，规则清晰易懂</li>
                <li>质量优秀，接近原神经网络性能</li>
                <li>人工可编辑，支持规则调整</li>
                <li>部署友好，符号算法易于实现</li>
                <li>信任度高，可解释性增强用户信任</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>转换过程复杂，需要专门的技术</li>
                <li>可能损失少量性能</li>
                <li>转换质量依赖原模型的质量</li>
                <li>符号规则可能过于复杂</li>
            </ul>
        """,
        "applications": """
            <p>N2S特别适合：</p>
            <ul>
                <li>需要可解释性的关键应用</li>
                <li>需要人工审查和验证的场景</li>
                <li>需要将神经网络知识迁移到传统系统的项目</li>
                <li>研究和教学，展示神经符号结合</li>
            </ul>
        """
    },
    "NeuOpt": {
        "name": "NeuOpt",
        "full_name": "Neural Optimizer",
        "category": "改进方法",
        "year": "2023",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>NeuOpt 使用神经网络学习优化算法本身，而不是学习问题的解，实现元学习层面的优化策略学习。</p>
            <div class="highlight-box">
                <strong>核心创新</strong>：将优化算法参数化，使用神经网络根据问题特征动态生成最优的优化策略，实现算法级别的自适应。
            </div>
        """,
        "architecture": """
            <h4>1. 算法参数化</h4>
            <ul>
                <li>将优化算法的关键参数表示为可学习函数</li>
                <li>设计算法模板，参数由神经网络生成</li>
                <li>支持多种优化算法类型</li>
            </ul>

            <h4>2. 元学习网络</h4>
            <ul>
                <li>输入问题特征，输出优化算法参数</li>
                <li>学习问题特征到算法参数的映射</li>
                <li>支持快速适应新问题</li>
            </ul>

            <h4>3. 双层优化</h4>
            <ul>
                <li>内层：使用生成的算法优化问题</li>
                <li>外层：优化元学习网络参数</li>
                <li>端到端训练整个系统</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>算法学习</strong>：学习优化算法而非解</li>
                <li>🔹 <strong>元学习</strong>：快速适应新问题</li>
                <li>🔹 <strong>自适应</strong>：根据问题特征调整算法</li>
                <li>🔹 <strong>通用性</strong>：一个框架学习多种算法</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>适应性</h5>
                    <p>快速适应<br>新问题类型</p>
                </div>
                <div class="info-card">
                    <h5>性能</h5>
                    <p>接近手工设计<br>算法性能</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">NeuOpt能够学习到接近手工设计算法的性能，同时具有更强的适应性。</p>
        """,
        "advantages": """
            <ul>
                <li>学习能力强，能自动发现优化策略</li>
                <li>适应性好，快速适应新问题</li>
                <li>通用性强，一个框架学习多种算法</li>
                <li>自动化程度高，减少人工设计</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>训练复杂，需要双层优化</li>
                <li>泛化挑战，可能过拟合训练问题</li>
                <li>计算开销大，需要多次迭代优化</li>
                <li>可解释性差，学习的算法难以理解</li>
            </ul>
        """,
        "applications": """
            <p>NeuOpt特别适合：</p>
            <ul>
                <li>需要快速适应新问题类型的场景</li>
                <li>问题类型多样但结构相似的应用</li>
                <li>自动化算法设计的研究</li>
                <li>元学习和自动机器学习的研究</li>
            </ul>
        """
    },
    "ActiveSearch": {
        "name": "ActiveSearch",
        "full_name": "Active Search",
        "category": "传导式强化学习",
        "year": "2020",
        "paper_url": "https://arxiv.org/abs/2012.05417",
        "paper_venue": "ICLR 2021",
        "core_concept": """
            <p>Active Search 在测试时继续优化策略，通过主动搜索和策略更新来改进解质量，是一种传导式（transductive）学习方法。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：不同于传统方法训练后固定策略，Active Search在测试时针对每个具体问题实例继续优化策略，通过多次迭代搜索找到更好的解。
            </div>
        """,
        "architecture": """
            <h4>1. 测试时策略更新</h4>
            <ul>
                <li>在测试实例上继续训练策略网络</li>
                <li>使用REINFORCE等算法更新参数</li>
                <li>针对当前问题实例优化</li>
            </ul>

            <h4>2. 主动搜索机制</h4>
            <ul>
                <li>生成多个候选解</li>
                <li>评估每个解的质量</li>
                <li>选择最优解或继续优化</li>
            </ul>

            <h4>3. 迭代优化</h4>
            <ul>
                <li>多次迭代策略更新</li>
                <li>逐步改进解质量</li>
                <li>直到达到时间限制或收敛</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>测试时优化</strong>：在测试阶段继续学习</li>
                <li>🔹 <strong>传导式学习</strong>：利用测试实例信息</li>
                <li>🔹 <strong>质量提升</strong>：显著改善解的质量</li>
                <li>🔹 <strong>实例特定</strong>：针对每个问题实例优化</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>质量提升</h5>
                    <p>Gap降低: 30-50%<br>vs 固定策略</p>
                </div>
                <div class="info-card">
                    <h5>TSP-50</h5>
                    <p>Gap: ~0.3%<br>Time: 1-5min</p>
                </div>
                <div class="info-card">
                    <h5>迭代次数</h5>
                    <p>100-1000次<br>逐步优化</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">Active Search通过测试时优化，能够显著提升解的质量，但需要较长的计算时间。</p>
        """,
        "advantages": """
            <ul>
                <li>测试时优化，针对每个实例定制</li>
                <li>质量提升显著，Gap降低30-50%</li>
                <li>灵活性高，可以适应不同问题</li>
                <li>不需要重新训练模型</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>推理时间长，需要多次迭代</li>
                <li>计算资源需求高，需要GPU支持</li>
                <li>不适合实时应用</li>
                <li>每次测试都需要优化，无法复用</li>
            </ul>
        """,
        "applications": """
            <p>Active Search特别适合：</p>
            <ul>
                <li>对解质量要求极高的场景</li>
                <li>可以接受较长计算时间的应用</li>
                <li>离线优化和批量处理</li>
                <li>学术研究和性能对比</li>
            </ul>
        """
    },
    "EAS": {
        "name": "EAS",
        "full_name": "Efficient Active Search",
        "category": "传导式强化学习",
        "year": "2021",
        "paper_url": "https://arxiv.org/abs/2106.05126",
        "paper_venue": "NeurIPS 2021",
        "core_concept": """
            <p>EAS 是Active Search的高效版本，通过多种优化技术减少测试时优化的计算开销，在保持解质量的同时显著提升效率。</p>
            <div class="highlight-box">
                <strong>核心改进</strong>：通过早停策略、梯度近似、批量优化等技术，将Active Search的计算时间减少5-10倍，同时保持接近的性能。
            </div>
        """,
        "architecture": """
            <h4>1. 早停策略</h4>
            <ul>
                <li>监控优化进度，提前终止无望的搜索</li>
                <li>设置收敛阈值，避免过度优化</li>
                <li>动态调整迭代次数</li>
            </ul>

            <h4>2. 高效梯度估计</h4>
            <ul>
                <li>使用近似梯度而非精确梯度</li>
                <li>减少前向和反向传播次数</li>
                <li>批量处理多个候选解</li>
            </ul>

            <h4>3. 智能采样</h4>
            <ul>
                <li>优先采样有希望的候选解</li>
                <li>减少无效搜索</li>
                <li>提高搜索效率</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>效率优化</strong>：计算时间减少5-10倍</li>
                <li>🔹 <strong>早停策略</strong>：避免无效的继续优化</li>
                <li>🔹 <strong>梯度近似</strong>：快速估计策略梯度</li>
                <li>🔹 <strong>质量保持</strong>：接近Active Search的性能</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>速度提升</h5>
                    <p>比AS快: 5-10x<br>效率优化</p>
                </div>
                <div class="info-card">
                    <h5>TSP-50</h5>
                    <p>Gap: ~0.4%<br>Time: 10-30s</p>
                </div>
                <div class="info-card">
                    <h5>质量保持</h5>
                    <p>95-98%<br>vs Active Search</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">EAS在保持接近Active Search性能的同时，显著提升了计算效率。</p>
        """,
        "advantages": """
            <ul>
                <li>更快的测试时优化，速度提升5-10倍</li>
                <li>效率与质量平衡好，性能损失小</li>
                <li>计算资源需求降低</li>
                <li>更适合实际应用</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>仍需额外计算，比固定策略慢</li>
                <li>可能错过最优解（早停导致）</li>
                <li>需要调整早停和近似参数</li>
                <li>仍不适合实时应用</li>
            </ul>
        """,
        "applications": """
            <p>EAS特别适合：</p>
            <ul>
                <li>需要高质量解但时间有限的应用</li>
                <li>批量处理问题的场景</li>
                <li>对质量和效率都有要求的项目</li>
                <li>实际生产环境的优化任务</li>
            </ul>
        """
    }
}

# ============================================
    # 构造方法（自回归）
    # ============================================
    
    # TODO: 将 app.py 中第 718-1809 行的 MODEL_DATABASE 内容移动到这里
    # 包含以下模型：
    # - AM (Attention Model)
    # - POMO (Policy Optimization with Multiple Optima)
    # - SymNCO (Symmetric Neural Combinatorial Optimization)
    # - MatNet (Matrix Network)
    # - HAM (Hierarchical Attention Model)
    # - PolyNet (Polynomial Time Network)
    # - MTPOMO (Multi-Task POMO)
    # - MVMoE (Multi-View Mixture of Experts)
    # - L2D (Learn to Delegate)
    # - HGNN (Heterogeneous Graph Neural Network)
    # - DF (Distribution Fitting)
    # - PtrNet (Pointer Network)
    
    # ============================================
    # 构造方法（非自回归）
    # ============================================
    
    # - DeepACO (Deep Ant Colony Optimization)
    # - GFACS (Graph-based Fast Adaptive Construction Solver)
    # - GLOP (Generalized Learning for Optimization Problems)
    
    # ============================================
    # 强化学习算法
    # ============================================
    
    # - REINFORCE
    # - A2C (Advantage Actor-Critic)
    # - PPO (Proximal Policy Optimization)
    
    # ============================================
    # 改进方法
    # ============================================
    
    # - DACT (Dual Attention with Cross Transformation)
    # - N2S (Neural to Symbolic)
    # - NeuOpt (Neural Optimizer)
    
    # ============================================
    # 传导式强化学习
    # ============================================
    
    # - ActiveSearch (Active Search)
    # - EAS (Efficient Active Search)



def get_model_categories():
    """
    获取所有模型分类
    
    Returns:
        set: 所有不重复的分类名称集合
    """
    categories = set()
    for model in MODEL_DATABASE.values():
        if 'category' in model:
            categories.add(model['category'])
    return categories


def get_model_by_id(model_id):
    """
    根据模型ID获取模型信息
    
    Args:
        model_id (str): 模型ID（如 'AM', 'POMO'）
    
    Returns:
        dict or None: 模型信息字典，如果不存在则返回None
    """
    return MODEL_DATABASE.get(model_id)


def get_models_by_category(category):
    """
    获取指定分类下的所有模型
    
    Args:
        category (str): 分类名称
    
    Returns:
        dict: 该分类下的所有模型字典
    """
    return {
        model_id: model_info 
        for model_id, model_info in MODEL_DATABASE.items() 
        if model_info.get('category') == category
    }


def search_models(keyword):
    """
    搜索模型（根据名称、全称或核心概念）
    
    Args:
        keyword (str): 搜索关键词
    
    Returns:
        dict: 匹配的模型字典
    """
    keyword_lower = keyword.lower()
    results = {}
    
    for model_id, model_info in MODEL_DATABASE.items():
        # 搜索模型名称
        if keyword_lower in model_info.get('name', '').lower():
            results[model_id] = model_info
            continue
        
        # 搜索完整名称
        if keyword_lower in model_info.get('full_name', '').lower():
            results[model_id] = model_info
            continue
        
        # 搜索核心概念
        if keyword_lower in model_info.get('core_concept', '').lower():
            results[model_id] = model_info
            continue
    
    return results

