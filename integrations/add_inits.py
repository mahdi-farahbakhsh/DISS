import os

def add_init_files(root_dir):
    if not os.path.exists(root_dir):
        print(f"Directory '{root_dir}' does not exist. Skipping.")
        return

    for dirpath, dirnames, filenames in os.walk(root_dir):
        init_file = os.path.join(dirpath, '__init__.py')
        if not os.path.exists(init_file):
            open(init_file, 'a').close()
            print(f'Created: {init_file}')

for dir in ['integrations/dps/diffusion-posterior-sampling', 'hello']:
    add_init_files(dir)
