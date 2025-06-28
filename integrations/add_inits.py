import os

def contains_python_file(dirpath):
    for root, _, files in os.walk(dirpath):
        if any(f.endswith('.py') for f in files):
            return True
    return False

def add_init_files(root_dir):
    if not os.path.exists(root_dir):
        print(f"Directory '{root_dir}' does not exist. Skipping.")
        return

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip hidden directories
        if any(part.startswith('.') for part in dirpath.split(os.sep)):
            continue

        # Only proceed if the directory or its subdirs contain a .py file
        if contains_python_file(dirpath):
            init_file = os.path.join(dirpath, '__init__.py')
            if not os.path.exists(init_file):
                open(init_file, 'a').close()
                print(f'Created: {init_file}')


# List of root directories to scan
for dir in [
    'dps/diffusion-posterior-sampling'
]:
    add_init_files(dir)
