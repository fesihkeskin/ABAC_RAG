import os

def generate_tree(root_dir, prefix="", exclude=None, output_file=None):
    """
    Recursively generates the folder structure with comments.
    Excludes files and directories based on the 'exclude' list.
    """
    if exclude is None:
        exclude = []

    files = os.listdir(root_dir)
    # Filter to exclude hidden files, folders, and items in the exclude list
    files = [f for f in files if not f.startswith('.') and not any(
        f == ex or f.endswith(ex) for ex in exclude)]
    
    entries = [os.path.join(root_dir, f) for f in files]
    # Sort directories first
    entries.sort(key=lambda x: (not os.path.isdir(x), x))
    
    for i, entry in enumerate(entries):
        is_last = i == (len(entries) - 1)
        more = '└──' if is_last else '├──'
        line = f"{prefix}{more} {os.path.basename(entry)}/" if os.path.isdir(entry) else f"{prefix}{more} {os.path.basename(entry)}"
        
        if output_file:
            output_file.write(line + "\n")
        else:
            print(line)
        
        # default_comment = '# Add comments here'
        default_comment = ' '
        # Print or write comment space if it's a folder
        if os.path.isdir(entry):
            comment_line = f"{prefix}{'    ' if is_last else '│   '}{default_comment}"
            if output_file:
                output_file.write(comment_line + "\n")
            else:
                print(comment_line)
            
            new_prefix = prefix + ('    ' if is_last else '│   ')
            generate_tree(entry, new_prefix, exclude, output_file)

# Example usage
project_root = os.getcwd()
print(f'Current working directory: {project_root}')

# Add folders or file extensions to exclude
# exclude_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','__pycache__', '.git', '.DS_Store', '.pyc', '.pkl', '.json', '.csv', '.txt']  
exclude_list = ['.jpg','.JPG','.jpeg','.JPEG','__pycache__', '.git', '.DS_Store', '.pyc', '.pkl']  

print(f"{os.path.basename(project_root)}/")
print("# Root of the project")
generate_tree(project_root, exclude=exclude_list)

saving_root = os.path.join(project_root, 'references', 'project_structure.txt')
print(f"Saving to: {saving_root}")

# Writing the tree structure to a text file
with open(saving_root, 'w', encoding='utf-8') as f:
    f.write(f"{os.path.basename(project_root)}/\n")
    f.write("# Root of the project\n")
    generate_tree(project_root, exclude=exclude_list, output_file=f)

print(f"Directory tree saved to {saving_root}")
