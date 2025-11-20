import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque, defaultdict

# -----------------------
# Visual / theme helpers
# -----------------------

def local_css(css):
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

BASE_CSS = """
/* page background */
.reportview-container .main {background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);} 
/* card like panels */
.card {background: white; border-radius: 14px; box-shadow: 0 6px 18px rgba(15,23,42,0.08); padding: 14px;}
.header {font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;}
.small-muted {color: #6b7280; font-size:0.9rem}
.stat {font-size:1.35rem; font-weight:700}
.badge {display:inline-block; padding:6px 10px; border-radius:999px; font-weight:600}
.btn-row {display:flex; gap:8px}
"""

# -----------------------
# Environment and helpers
# -----------------------

ACTION_LIST = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
ACTION_DELTA = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
    "STAY": (0, 0),
}


def clamp_pos(pos, rows, cols):
    r, c = pos
    r = max(0, min(rows - 1, r))
    c = max(0, min(cols - 1, c))
    return (r, c)

class GridEnvironment:
    def __init__(self, rows=6, cols=6, dirt_prob=0.2, seed=None):
        self.rows = rows
        self.cols = cols
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.reset(dirt_prob)

    def reset(self, dirt_prob=0.2):
        # 1 = dirt, 0 = clean
        self.dirt = (np.random.rand(self.rows, self.cols) < dirt_prob).astype(int)
        # Choose agent start randomly
        empty_cells = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        self.agent_pos = random.choice(empty_cells)
        self.time = 0
        self.total_cleaned = 0
        self.history = []  # record remaining dirt over time

    def is_dirty(self, pos):
        r, c = pos
        return bool(self.dirt[r, c])

    def clean(self, pos):
        r, c = pos
        if self.dirt[r, c] == 1:
            self.dirt[r, c] = 0
            self.total_cleaned += 1
            return True
        return False

    def step(self, action):
        dr, dc = ACTION_DELTA[action]
        newpos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
        newpos = clamp_pos(newpos, self.rows, self.cols)
        self.agent_pos = newpos
        cleaned = False
        if self.is_dirty(self.agent_pos):
            cleaned = self.clean(self.agent_pos)
        self.time += 1
        self.history.append(int(self.dirt.sum()))
        return {
            "pos": self.agent_pos,
            "cleaned": cleaned,
            "time": self.time,
            "total_cleaned": self.total_cleaned,
            "remaining_dirt": int(self.dirt.sum()),
        }

# -----------------------
# Agent base and subclasses
# -----------------------

class AgentBase:
    def __init__(self, env: GridEnvironment):
        self.env = env

    def perceive(self):
        return {"pos": self.env.agent_pos, "is_dirty": self.env.is_dirty(self.env.agent_pos)}

    def act(self):
        raise NotImplementedError

class SimpleReflexAgent(AgentBase):
    def act(self):
        p = self.perceive()
        if p["is_dirty"]:
            return "STAY"
        else:
            return random.choice(ACTION_LIST[:-1])

class ModelBasedReflexAgent(AgentBase):
    def __init__(self, env):
        super().__init__(env)
        self.cleaned_map = set()

    def act(self):
        pos = self.env.agent_pos
        if self.env.is_dirty(pos):
            self.cleaned_map.add(pos)
            return "STAY"
        neighbors = []
        for a in ["UP", "DOWN", "LEFT", "RIGHT"]:
            dr, dc = ACTION_DELTA[a]
            np_ = clamp_pos((pos[0] + dr, pos[1] + dc), self.env.rows, self.env.cols)
            neighbors.append((a, np_))
        dirty_neighbors = [a for a, np_ in neighbors if self.env.is_dirty(np_)]
        if dirty_neighbors:
            return random.choice(dirty_neighbors)
        unseen = [a for a, np_ in neighbors if np_ not in self.cleaned_map]
        if unseen:
            return random.choice(unseen)
        return random.choice(ACTION_LIST[:-1])

class GoalBasedAgent(AgentBase):
    def __init__(self, env):
        super().__init__(env)
        self.path = deque()

    def find_nearest_dirty(self):
        start = self.env.agent_pos
        rows, cols = self.env.rows, self.env.cols
        visited = set([start])
        q = deque([(start, [])])
        while q:
            (r, c), path = q.popleft()
            if self.env.dirt[r, c] == 1:
                return path
            for a in ["UP", "DOWN", "LEFT", "RIGHT"]:
                dr, dc = ACTION_DELTA[a]
                np_ = clamp_pos((r + dr, c + dc), rows, cols)
                if np_ not in visited:
                    visited.add(np_)
                    q.append((np_, path + [a]))
        return None

    def act(self):
        if self.env.is_dirty(self.env.agent_pos):
            return "STAY"
        if not self.path:
            path = self.find_nearest_dirty()
            if path:
                for a in path:
                    self.path.append(a)
            else:
                return random.choice(ACTION_LIST[:-1])
        if self.path:
            return self.path.popleft()
        return random.choice(ACTION_LIST[:-1])

class UtilityBasedAgent(AgentBase):
    def __init__(self, env, energy_weight=0.5, cleanliness_weight=0.5):
        super().__init__(env)
        self.energy_weight = energy_weight
        self.cleanliness_weight = cleanliness_weight

    def estimated_utility(self, action):
        pos = clamp_pos((self.env.agent_pos[0] + ACTION_DELTA[action][0],
                          self.env.agent_pos[1] + ACTION_DELTA[action][1]),
                         self.env.rows, self.env.cols)
        will_clean = 1 if self.env.is_dirty(pos) else 0
        energy_cost = 0 if action == "STAY" else 1
        utility = self.cleanliness_weight * will_clean - self.energy_weight * energy_cost
        return utility

    def act(self):
        best = None
        best_val = -1e9
        for a in ACTION_LIST:
            v = self.estimated_utility(a)
            if v > best_val:
                best_val = v
                best = [a]
            elif v == best_val:
                best.append(a)
        return random.choice(best)

class LearningAgent(AgentBase):
    def __init__(self, env, alpha=0.5, gamma=0.9, epsilon=0.2):
        super().__init__(env)
        if "q_table" not in st.session_state:
            st.session_state.q_table = defaultdict(lambda: {a: 0.0 for a in ACTION_LIST})
        self.q = st.session_state.q_table
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.last_state = None
        self.last_action = None

    def state_key(self):
        # richer state: (r,c, local dirt bitmask of 3x3 around agent)
        r, c = self.env.agent_pos
        bits = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                rr, cc = clamp_pos((r + dr, c + dc), self.env.rows, self.env.cols)
                bits.append(str(int(self.env.dirt[rr, cc])))
        key = (r, c, "".join(bits))
        return key

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTION_LIST)
        action_vals = self.q[state]
        max_val = max(action_vals.values())
        best_actions = [a for a, v in action_vals.items() if v == max_val]
        return random.choice(best_actions)

    def reward(self, cleaned):
        return 10 if cleaned else -1

    def act(self):
        s = self.state_key()
        if s not in self.q:
            self.q[s] = {a: 0.0 for a in ACTION_LIST}
        a = self.choose_action(s)
        self.last_state = s
        self.last_action = a
        return a

    def learn(self, cleaned):
        if self.last_state is None or self.last_action is None:
            return
        s = self.last_state
        a = self.last_action
        r = self.reward(cleaned)
        s2 = self.state_key()
        if s2 not in self.q:
            self.q[s2] = {a: 0.0 for a in ACTION_LIST}
        max_next = max(self.q[s2].values())
        self.q[s][a] = self.q[s][a] + self.alpha * (r + self.gamma * max_next - self.q[s][a])
        self.last_state = None
        self.last_action = None

# -----------------------
# Streamlit UI + main
# -----------------------

st.set_page_config(layout="wide", page_title="Types of Agents", page_icon="🤖")
local_css(BASE_CSS)

st.markdown("""
<div style='display:flex;align-items:center;gap:16px'>
  <div style='flex:1'>
    <h1 class='header'>🤖 Types of Agents — Interactive Prototype</h1>
    <div class='small-muted'>A colorful Streamlit demo that visualizes Simple Reflex, Model-Based, Goal-Based, Utility, and Learning agents on a grid.</div>
  </div>
  <div style='text-align:right'>
    <div class='badge' style='background:linear-gradient(90deg,#7c3aed,#06b6d4); color:white'>Demo Mode</div>
  </div>
</div>
""", unsafe_allow_html=True)

# -----------------------
# Sidebar form (fixed)
# -----------------------
with st.sidebar.form("controls"):
    st.sidebar.header("Environment & Agent")
    rows = st.sidebar.slider("Grid rows", 4, 14, 6)
    grid_cols = st.sidebar.slider("Grid cols", 4, 14, 6)  # renamed to avoid collision
    dirt_prob = st.sidebar.slider("Dirt probability", 0.0, 0.8, 0.25, 0.05)
    agent_type = st.sidebar.selectbox("Agent type", [
        "Simple Reflex", "Model-Based Reflex", "Goal-Based", "Utility-Based", "Learning (Q-Learning)"
    ])
    seed_input = st.sidebar.number_input("Random seed (0=random)", value=0, step=1)
    steps_to_run = st.sidebar.number_input("Run for N steps", min_value=1, max_value=1000, value=10, step=1)
    st.sidebar.markdown("---")
    st.sidebar.write("Learning agent settings")
    alpha = st.sidebar.slider("Alpha (learning rate)", 0.0, 1.0, 0.5)
    gamma = st.sidebar.slider("Gamma (discount)", 0.0, 1.0, 0.9)
    epsilon = st.sidebar.slider("Epsilon (exploration)", 0.0, 1.0, 0.25)
    energy_weight = st.sidebar.slider("Energy weight (utility)", 0.0, 1.0, 0.4)
    cleanliness_weight = st.sidebar.slider("Cleanliness weight (utility)", 0.0, 1.0, 0.6)
    apply = st.form_submit_button("Apply / Create")

# convert seed 0 -> None for randomness
seed = None if seed_input == 0 else int(seed_input)

# initialize environment in session state (on first load or when user presses Apply)
if "env" not in st.session_state or apply:
    st.session_state.env = GridEnvironment(rows=rows, cols=grid_cols, dirt_prob=dirt_prob, seed=seed)
    st.session_state.agent_obj = None
    st.session_state.logs = []
    st.session_state.episode = 0

env = st.session_state.env

# create agent object factory
def create_agent_for_type(agent_type_name):
    if agent_type_name == "Simple Reflex":
        return SimpleReflexAgent(env)
    elif agent_type_name == "Model-Based Reflex":
        return ModelBasedReflexAgent(env)
    elif agent_type_name == "Goal-Based":
        return GoalBasedAgent(env)
    elif agent_type_name == "Utility-Based":
        return UtilityBasedAgent(env, energy_weight=energy_weight, cleanliness_weight=cleanliness_weight)
    elif agent_type_name == "Learning (Q-Learning)":
        return LearningAgent(env, alpha=alpha, gamma=gamma, epsilon=epsilon)
    else:
        return SimpleReflexAgent(env)

# recreate agent when type changes or env changes
if "agent_type_prev" not in st.session_state or st.session_state.get("agent_type_prev") != agent_type:
    st.session_state.agent_obj = create_agent_for_type(agent_type)
    st.session_state.agent_type_prev = agent_type

if st.session_state.agent_obj is None or st.session_state.agent_obj.env is not env:
    st.session_state.agent_obj = create_agent_for_type(agent_type)

agent = st.session_state.agent_obj

# Top-level layout: left (visual), right (controls & stats)
left, right = st.columns([2, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Environment Visualization")
    fig, ax = plt.subplots(figsize=(5, 5))
    # nicer palette
    clean_color = np.array([0.98, 0.99, 1.0])
    dirt_color = np.array([0.85, 0.53, 0.33])
    grid_img = np.zeros((env.rows, env.cols, 3), dtype=float) + clean_color
    dirt_mask = env.dirt == 1
    grid_img[dirt_mask] = dirt_color
    ax.imshow(grid_img, interpolation="nearest")
    ar = env.agent_pos
    # draw agent halo
    ax.scatter([ar[1]], [ar[0]], s=700, facecolors='none', edgecolors=(0.1,0.2,0.6,0.12), linewidths=12)
    ax.scatter([ar[1]], [ar[0]], s=250, c=(0.15,0.45,0.95), marker='o', edgecolors='k')
    # grid lines
    ax.set_xticks(np.arange(-0.5, env.cols, 1))
    ax.set_yticks(np.arange(-0.5, env.rows, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)
    ax.set_title(f"Grid {env.rows}×{env.cols} — Dirt left: {int(env.dirt.sum())}")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Agent & Controls")
    st.markdown(f"**Agent type:** <span class='small-muted'>{agent_type}</span>", unsafe_allow_html=True)
    st.markdown(f"**Position:** <span class='stat'>{env.agent_pos}</span>", unsafe_allow_html=True)
    st.markdown(f"**Time:** <span class='stat'>{env.time}</span>", unsafe_allow_html=True)
    st.markdown(f"**Total cleaned:** <span class='stat'>{env.total_cleaned}</span>", unsafe_allow_html=True)
    st.markdown(f"**Remaining dirt:** <span class='stat'>{int(env.dirt.sum())}</span>", unsafe_allow_html=True)
    st.markdown("---")

    # Step / run / reset buttons (use names that do not conflict)
    btn_cols = st.columns([1,1,1])
    if btn_cols[0].button("🔁 Step once"):
        action = agent.act()
        res = env.step(action)
        cleaned = res['cleaned']
        if isinstance(agent, LearningAgent):
            agent.learn(cleaned)
        if isinstance(agent, ModelBasedReflexAgent) and cleaned:
            agent.cleaned_map.add(env.agent_pos)
        st.session_state.logs.append(f"t={env.time}: {action} ➜ pos={env.agent_pos} cleaned={cleaned}")

    if btn_cols[1].button("▶️ Run N steps"):
        for _ in range(steps_to_run):
            if env.dirt.sum() == 0:
                st.session_state.logs.append(f"All clean at t={env.time}; stopped.")
                break
            action = agent.act()
            res = env.step(action)
            cleaned = res['cleaned']
            if isinstance(agent, LearningAgent):
                agent.learn(cleaned)
            if isinstance(agent, ModelBasedReflexAgent) and cleaned:
                agent.cleaned_map.add(env.agent_pos)
            st.session_state.logs.append(f"t={env.time}: {action} ➜ pos={env.agent_pos} cleaned={cleaned}")

    if btn_cols[2].button("🧹 Reset"):
        st.session_state.env = GridEnvironment(rows=rows, cols=grid_cols, dirt_prob=dirt_prob, seed=seed)
        st.session_state.agent_obj = create_agent_for_type(agent_type)
        st.session_state.logs = []
        st.session_state.episode += 1
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("Recent activity")
    logs = st.session_state.get("logs", [])
    for log in logs[-6:][::-1]:
        st.write(log)
    st.markdown("</div>", unsafe_allow_html=True)

# bottom: analytics row
st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
col_a, col_b = st.columns([1,1])
with col_a:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Cleaning progress")
    # show small line chart of remaining dirt over time
    if len(env.history) > 1:
        fig2, ax2 = plt.subplots(figsize=(4,2.2))
        ax2.plot(range(len(env.history)), env.history, marker='o')
        ax2.set_xlabel('steps')
        ax2.set_ylabel('remaining dirt')
        ax2.set_title('Remaining dirt over time')
        ax2.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.info('Run a few steps to see progress chart')
    st.markdown("</div>", unsafe_allow_html=True)

with col_b:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Agent Q-table sample / utilities")
    if isinstance(agent, LearningAgent):
        q_table = st.session_state.q_table
        sample_keys = list(q_table.keys())[:6]
        if not sample_keys:
            st.write('Q-table empty — let the agent explore!')
        else:
            for k in sample_keys:
                st.write(f"state={k} → { {a: round(v,2) for a,v in q_table[k].items()} }")
    elif isinstance(agent, UtilityBasedAgent):
        st.write('Utility agent weights:')
        st.write(f'energy_weight={agent.energy_weight}, cleanliness_weight={agent.cleanliness_weight}')
    else:
        st.write('Switch to Learning or Utility agent to see internal values.')
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("Tip: Use the sidebar to change grid size, dirt density, agent type and learning params. This UI focuses on clarity and nicer visuals for presentations.")
