import json
import re

def split_notebook(input_file, output_prefix):
    """Split the notebook into three separate notebooks based on analysis types."""
    
    # Read the original notebook
    with open(input_file, 'r') as f:
        notebook = json.load(f)
    
    # Define the three analysis sections
    sections = {
        'raw_values': {
            'start_marker': '# Fire Weather Index (Raw Values) Analysis',
            'end_marker': '# High Fire Danger Frequency Analysis',
            'title': 'FWI_Raw_Values_Analysis'
        },
        'high_fire_danger': {
            'start_marker': '# High Fire Danger Frequency Analysis',
            'end_marker': '# 95th Percentile Analysis',
            'title': 'FWI_High_Fire_Danger_Analysis'
        },
        'percentile': {
            'start_marker': '# 95th Percentile Analysis',
            'end_marker': None,  # End of notebook
            'title': 'FWI_95th_Percentile_Analysis'
        }
    }
    
    # Extract common setup cells (first few cells before any analysis)
    setup_cells = []
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if any(marker in source for marker in ['# Fire Weather Index (Raw Values) Analysis', 
                                                  '# High Fire Danger Frequency Analysis', 
                                                  '# 95th Percentile Analysis']):
                break
            setup_cells.append(cell)
        else:
            setup_cells.append(cell)
    
    # Create three separate notebooks
    for section_name, section_info in sections.items():
        print(f"Creating {section_info['title']}.ipynb...")
        
        # Start with setup cells
        new_cells = setup_cells.copy()
        
        # Find cells for this section
        in_section = False
        section_cells = []
        
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell.get('source', []))
                
                # Check if we're entering this section
                if section_info['start_marker'] in source:
                    in_section = True
                    section_cells.append(cell)
                    continue
                
                # Check if we're leaving this section
                if in_section and section_info['end_marker'] and section_info['end_marker'] in source:
                    in_section = False
                    break
                
                # Add cells while in section
                if in_section:
                    section_cells.append(cell)
        
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
