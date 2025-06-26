import os
import csv
import datetime


def list_existing_agents(base_dir="runs"):
    """Lists agent directories that are not old-style 'vX_...' runs, sorted alphabetically."""
    if not os.path.exists(base_dir):
        return []
    agents = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('v')]
    agents.sort()  # Sort alphabetically
    return agents


def get_agent_path(agent_name, base_dir="runs"):
    """Returns the full path to an agent's directory."""
    return os.path.join(base_dir, agent_name)


def get_next_session_dir(agent_path):
    """
    Creates a new versioned session directory within an agent's folder.
    Increments the session number based on existing 'session_Y' directories and adds a timestamp.
    Example: runs/MyAgent/session_1_20231027-103000
    """
    if not os.path.exists(agent_path):
        os.makedirs(agent_path)

    existing_sessions = [d for d in os.listdir(agent_path) if os.path.isdir(os.path.join(agent_path, d)) and d.startswith('session_')]

    next_session = 1
    if existing_sessions:
        session_numbers = []
        for d in existing_sessions:
            parts = d.split('_')
            # Check for 'session_N' format and N is digit
            if len(parts) > 1 and parts[0] == 'session' and parts[1].isdigit():
                session_numbers.append(int(parts[1]))
        if session_numbers:
            next_session = max(session_numbers) + 1

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = os.path.join(agent_path, f"session_{next_session}_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def find_latest_session_path(agent_path):
    """
    Finds the directory of the latest training session for a given agent.
    Assumes session directories are named 'session_Y_YYYYMMDD-HHMMSS'.
    """
    if not os.path.exists(agent_path):
        return None

    session_dirs = [d for d in os.listdir(agent_path) if os.path.isdir(os.path.join(agent_path, d)) and d.startswith('session_')]

    if not session_dirs:
        return None

    # Sort by parsing the session number and then the timestamp for robust ordering
    def sort_key(dir_name):
        parts = dir_name.split('_')
        session_num = 0
        timestamp_str = ""
        if len(parts) > 1 and parts[0] == 'session' and parts[1].isdigit():
            session_num = int(parts[1])
        if len(parts) > 2:  # Check for timestamp part
            timestamp_str = parts[2]
        return session_num, timestamp_str

    session_dirs.sort(key=sort_key)
    latest_session_dir_name = session_dirs[-1]
    latest_session_path = os.path.join(agent_path, latest_session_dir_name)
    return latest_session_path


def find_sessions_for_agent(agent_path):
    """
    Finds all session directories for a given agent and returns them sorted.
    Assumes session directories are named 'session_Y_YYYYMMDD-HHMMSS'.
    """
    if not os.path.exists(agent_path):
        return []

    session_dirs = [d for d in os.listdir(agent_path) if os.path.isdir(os.path.join(agent_path, d)) and d.startswith('session_')]

    if not session_dirs:
        return []

    # Sort by parsing the session number and then the timestamp
    def sort_key(dir_name):
        parts = dir_name.split('_')
        session_num = 0
        timestamp_str = ""
        if len(parts) > 1 and parts[0] == 'session' and parts[1].isdigit():
            session_num = int(parts[1])
        if len(parts) > 2:
            timestamp_str = parts[2]
        return session_num, timestamp_str

    session_dirs.sort(key=sort_key)
    return [os.path.join(agent_path, s_dir) for s_dir in session_dirs]


def save_agent_config(agent_path, config_dict):
    """Saves agent specific configuration to a CSV file."""
    config_file = os.path.join(agent_path, "agent_config.csv")
    with open(config_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in config_dict.items():
            writer.writerow([key, str(value)])  # Store value as string
    print(f"Agent configuration saved to {config_file}")


def load_agent_config(agent_path):
    """Loads agent specific configuration from a CSV file."""
    config_file = os.path.join(agent_path, "agent_config.csv")
    config_dict = {}
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Agent configuration file not found: {config_file}")
    with open(config_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                try:
                    # Use eval to handle lists/tuples (hiddenDims) and numbers; strings will remain strings.
                    value = eval(row[1])
                except (NameError, SyntaxError):
                    value = row[1]
                config_dict[row[0]] = value
    print(f"Agent configuration loaded from {config_file}")
    return config_dict
