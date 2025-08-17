import json
import re

def find_cell_boundaries(notebook, target_lines):
    """Find the cell boundaries for given line numbers."""
    cell_boundaries = []
    current_line = 0
    
    for i, cell in enumerate(notebook['cells']):
        cell_lines = 0
        if 'source' in cell:
            cell_lines = len(''.join(cell['source']).split('\n'))
        
        # Check if any target line falls within this cell
        for target_line in target_lines:
            if current_line <= target_line < current_line + cell_lines:
                cell_boundaries.append(i)
                break
        
        current_line += cell_lines
    
    return cell_boundaries

def split_notebook(input_file, output_prefix):
    """Split the notebook into three separate notebooks based on analysis types."""
    
    # Read the original notebook
    with open(input_file, 'r') as f:
        notebook = json.load(f)
    
    # Find the line numbers where each analysis section starts
    notebook_text = json.dumps(notebook, indent=1)
    lines = notebook_text.split('\n')
    
    section_starts = {}
    for i, line in enumerate(lines):
        if '"# Fire Weather Index (Raw Values) Analysis"' in line:
            section_starts['raw_values'] = i
        elif '"# High Fire Danger Frequency Analysis"' in line:
            section_starts['high_fire_danger'] = i
        elif '"# 95th Percentile Analysis"' in line:
            section_starts['percentile'] = i
    
    print(f"Found section starts at lines: {section_starts}")
    
    # Define the sections
    sections = {
        'raw_values': {
            'start_line': section_starts['raw_values'],
            'end_line': section_starts['high_fire_danger'],
            'title': 'FWI_Raw_Values_Analysis'
        },
        'high_fire_danger': {
            'start_line': section_starts['high_fire_danger'],
            'end_line': section_starts['percentile'],
            'title': 'FWI_High_Fire_Danger_Analysis'
        },
        'percentile': {
            'start_line': section_starts['percentile'],
            'end_line': len(lines),
            'title': 'FWI_95th_Percentile_Analysis'
        }
    }
    
    # Extract common setup cells (cells before the first analysis section)
    setup_cells = []
    setup_end_line = section_starts['raw_values']
    
    current_line = 0
    for cell in notebook['cells']:
        cell_lines = 0
        if 'source' in cell:
            cell_lines = len(''.join(cell['source']).split('\n'))
        
        if current_line + cell_lines <= setup_end_line:
            setup_cells.append(cell)
        else:
            break
        
        current_line += cell_lines
    
    # Create three separate notebooks
    for section_name, section_info in sections.items():
        print(f"Creating {section_info['title']}.ipynb...")
        
        # Start with setup cells
        new_cells = setup_cells.copy()
        
        # Extract cells for this section
        section_cells = []
        current_line = 0
        
        for cell in notebook['cells']:
            cell_lines = 0
            if 'source' in cell:
                cell_lines = len(''.join(cell['source']).split('\n'))
            
            # Check if this cell falls within the section boundaries
            cell_start = current_line
            cell_end = current_line + cell_lines
            
            if (cell_start >= section_info['start_line'] and 
                cell_end <= section_info['end_line']):
                section_cells.append(cell)
            
            current_line += cell_lines
        
        # Add section cells to new notebook
        new_cells.extend(section_cells)
        
        # Create new notebook structure
        new_notebook = {
            'cells': new_cells,
            'metadata': notebook.get('metadata', {}),
            'nbformat': notebook.get('nbformat', 4),
            'nbformat_minor': notebook.get('nbformat_minor', 4)
        }
        
        # Write the new notebook
        output_file = f"{output_prefix}_{section_name}.ipynb"
        with open(output_file, 'w') as f:
            json.dump(new_notebook, f, indent=1)
        
        print(f"Created {output_file} with {len(new_cells)} cells")

if __name__ == "__main__":
    split_notebook('FWI_HFD_MultimodelMean_Significance.ipynb', 'FWI_Analysis')
