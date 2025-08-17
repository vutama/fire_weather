import json

def split_notebook(input_file, output_prefix):
    """Split the notebook into three separate notebooks based on analysis types."""
    
    # Read the original notebook
    with open(input_file, 'r') as f:
        notebook = json.load(f)
    
    # Find the cell indices where each analysis section starts (in markdown cells)
    section_starts = {}
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown' and 'source' in cell:
            source = ''.join(cell['source'])
            if '# Fire Weather Index (Raw Values) Analysis' in source:
                section_starts['raw_values'] = i
            elif '# High Fire Danger Frequency Analysis' in source:
                section_starts['high_fire_danger'] = i
            elif '# 95th Percentile Analysis' in source:
                section_starts['percentile'] = i
    
    print(f"Found section starts at cells: {section_starts}")
    
    # Define the sections
    sections = {
        'raw_values': {
            'start_cell': section_starts['raw_values'],
            'end_cell': section_starts['high_fire_danger'],
            'title': 'FWI_Raw_Values_Analysis'
        },
        'high_fire_danger': {
            'start_cell': section_starts['high_fire_danger'],
            'end_cell': section_starts['percentile'],
            'title': 'FWI_High_Fire_Danger_Analysis'
        },
        'percentile': {
            'start_cell': section_starts['percentile'],
            'end_cell': len(notebook['cells']),
            'title': 'FWI_95th_Percentile_Analysis'
        }
    }
    
    # Extract common setup cells (cells before the first analysis section)
    setup_cells = notebook['cells'][:section_starts['raw_values']]
    
    # Create three separate notebooks
    for section_name, section_info in sections.items():
        print(f"Creating {section_info['title']}.ipynb...")
        
        # Start with setup cells
        new_cells = setup_cells.copy()
        
        # Extract cells for this section
        section_cells = notebook['cells'][section_info['start_cell']:section_info['end_cell']]
        
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
