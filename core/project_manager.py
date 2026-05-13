import os
import json
import shutil
from datetime import datetime


class ProjectManager:
    def __init__(self):
        self.projects_dir = "projects"
        os.makedirs(self.projects_dir, exist_ok=True)

    def create_project(self, name):
        path = os.path.join(self.projects_dir, name)
        if os.path.exists(path):
            raise FileExistsError(f"Project '{name}' exists")

        os.makedirs(os.path.join(path, "datasets"), exist_ok=True)
        os.makedirs(os.path.join(path, "results"), exist_ok=True)

        self.save_state(name, {"created": datetime.now().isoformat()})

    def list_projects(self):
        return [
            d
            for d in os.listdir(self.projects_dir)
            if os.path.isdir(os.path.join(self.projects_dir, d))
        ]

    def save_state(self, project_name, state):
        path = os.path.join(self.projects_dir, project_name, "state.json")
        state["updated"] = datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def delete_project(self, name):
        path = os.path.join(self.projects_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Project '{name}' not found")

        shutil.rmtree(path)

    def load_state(self, project_name: str) -> dict:
        path = os.path.join(self.projects_dir, project_name, "state.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            return json.load(f)
