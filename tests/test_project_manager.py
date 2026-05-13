from core.project_manager import ProjectManager

pm = ProjectManager()

# CREATE PROJECT
project_path = pm.create_project("fraud_detection")

print("Project created:")

print(project_path)

# SAVE CONFIG
config = {"project": {"name": "fraud_detection"}}

config_path = pm.save_project_config("fraud_detection", config)

print("\nConfig saved:")

print(config_path)

# LOAD CONFIG
loaded_config = pm.load_project_config("fraud_detection")

print("\nLoaded Config:")

print(loaded_config)

# LIST PROJECTS
print("\nProjects:")

print(pm.list_projects())
