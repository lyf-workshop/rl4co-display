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
            <p>MVMoE 使用混合专家模型，针对不同类型的问题实例使用不同的专家网络。</p>
        """,
        "advantages": """
            <ul>
                <li>适应性强</li>
                <li>专业化处理</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>模型复杂</li>
                <li>训练困难</li>
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
            <p>L2D 学习何时使用神经网络求解，何时委托给传统启发式算法。</p>
        """,
        "advantages": """
            <ul>
                <li>灵活性高</li>
                <li>结合优势</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>决策复杂</li>
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
            <p>HGNN 使用异构图神经网络建模不同类型节点和边的关系。</p>
        """,
        "advantages": """
            <ul>
                <li>表达力强</li>
                <li>适用复杂问题</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>计算开销大</li>
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
            <p>DF 通过拟合最优解的分布来生成高质量解。</p>
        """,
        "advantages": """
            <ul>
                <li>理论优雅</li>
                <li>性能优秀</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>训练复杂</li>
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
            <p>GFACS 使用图神经网络非自回归地构建解。</p>
        """,
        "advantages": """
            <ul>
                <li>速度极快</li>
                <li>完全并行</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>质量可能不如自回归方法</li>
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
            <p>GLOP 是一个通用的学习框架，适用于多种优化问题。</p>
        """,
        "advantages": """
            <ul>
                <li>通用性强</li>
                <li>易于扩展</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>特定问题性能可能不如专用方法</li>
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
            <p>DACT 使用双重注意力机制和交叉变换来改进现有模型。</p>
        """,
        "advantages": """
            <ul>
                <li>即插即用</li>
                <li>性能提升明显</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>计算开销增加</li>
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
            <p>N2S 将神经网络的输出转换为符号化的优化算法。</p>
        """,
        "advantages": """
            <ul>
                <li>可解释性强</li>
                <li>质量优秀</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>转换过程复杂</li>
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
            <p>NeuOpt 使用神经网络学习优化算法本身。</p>
        """,
        "advantages": """
            <ul>
                <li>学习能力强</li>
                <li>适应性好</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>训练复杂</li>
                <li>泛化挑战</li>
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
            <p>Active Search 在测试时继续优化策略，通过主动搜索改进解质量。</p>
        """,
        "advantages": """
            <ul>
                <li>测试时优化</li>
                <li>质量提升显著</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>推理时间长</li>
                <li>计算资源需求高</li>
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
            <p>EAS 是Active Search的高效版本，减少了测试时优化的计算开销。</p>
        """,
        "advantages": """
            <ul>
                <li>更快的测试时优化</li>
                <li>效率与质量平衡好</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>仍需额外计算</li>
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

