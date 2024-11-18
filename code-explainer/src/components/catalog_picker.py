from dash import html, dcc, Input, Output, State
from databricks.sdk import WorkspaceClient
import dash_bootstrap_components as dbc
from ..services.code_analyzer import CodeAnalyzer
import dash_cytoscape as cyto
import os

class CatalogPicker:
    def __init__(self, app, workspace_client):
        self.app = app
        self.w = workspace_client
        self.code_analyzer = CodeAnalyzer(workspace_client)
        self.default_catalog = os.getenv('default_catalog', 'default')
        self.default_schema = os.getenv('default_schema', 'default')
        self.default_volume = os.getenv('default_volume', 'default')
        self.default_file = os.getenv('default_file', '')
        self._create_callbacks()

    def create_layout(self):
        try:
            catalog_options = [
                {'label': catalog.name, 'value': catalog.name} 
                for catalog in self.w.catalogs.list()
            ]
        except Exception as e:
            print(f'Error fetching catalogs: {str(e)}')
            catalog_options = []

        return html.Div([
            # File selection components
            html.Label('Catalog', className='fw-bold mb-1'),
            dcc.Dropdown(
                id='catalog-dropdown',
                options=catalog_options,
                value=self.default_catalog,
                placeholder='Select a catalog...',
                className='mb-3'
            ),
            
            html.Label('Schema', className='fw-bold mb-1'),
            dcc.Dropdown(
                id='schema-dropdown',
                options=[],
                placeholder='Select a schema...',
                className='mb-3'
            ),
            
            html.Label('Volume', className='fw-bold mb-1'),
            dcc.Dropdown(
                id='volume-dropdown',
                options=[],
                placeholder='Select a volume...',
                className='mb-3'
            ),
            
            html.Label('File', className='fw-bold mb-1'),
            dcc.Dropdown(
                id='file-picker-dropdown',
                options=[],
                placeholder='Select a file...',
                className='mb-3'
            ),
            
            # Parse button
            html.Div([
                dbc.Button(
                    'Parse File',
                    id='parse-file-button',
                    color='primary',
                    className='me-2',
                    style={'display': 'none'}
                ),
                html.Div(
                    id='parse-result',
                    style={'display': 'inline-block'}
                )
            ], className='d-flex align-items-center mb-3'),
        ], className='catalog-picker bg-white p-3 rounded shadow-sm')

    def _create_callbacks(self):
        @self.app.callback(
            Output('schema-dropdown', 'options'),
            Output('schema-dropdown', 'value'),
            Input('catalog-dropdown', 'value')
        )
        def update_schema_options(selected_catalog):
            if not selected_catalog:
                return [], None
            
            try:
                schemas = self.w.schemas.list(catalog_name=selected_catalog)
                schema_options = [
                    {'label': schema.name, 'value': schema.name}
                    for schema in schemas
                ]
                # Updated default schema logic
                default_schema = self.default_schema if selected_catalog == self.default_catalog else None
                return schema_options, default_schema
            except Exception as e:
                print(f'Error fetching schemas: {str(e)}')
                return [], None

        @self.app.callback(
            Output('volume-dropdown', 'options'),
            Output('volume-dropdown', 'value'),
            Input('schema-dropdown', 'value'),
            State('catalog-dropdown', 'value')
        )
        def update_volume_options(selected_schema, selected_catalog):
            if not selected_catalog or not selected_schema:
                return [], None
            
            try:
                volumes = self.w.volumes.list(
                    catalog_name=selected_catalog,
                    schema_name=selected_schema
                )
                volume_options = [
                    {'label': volume.name, 'value': volume.name}
                    for volume in volumes
                ]
                # Updated default volume logic
                default_volume = self.default_volume if (
                    selected_catalog == self.default_catalog and 
                    selected_schema == self.default_schema
                ) else None
                return volume_options, default_volume
            except Exception as e:
                print(f'Error fetching volumes: {str(e)}')
                return [], None

        @self.app.callback(
            Output('file-picker-dropdown', 'options'),
            Output('file-picker-dropdown', 'value'),
            Input('volume-dropdown', 'value'),
            State('catalog-dropdown', 'value'),
            State('schema-dropdown', 'value')
        )
        def update_file_options(selected_volume, selected_catalog, selected_schema):
            if not selected_catalog or not selected_schema or not selected_volume:
                return [], None
            
            try:
                volume_path = f'/Volumes/{selected_catalog}/{selected_schema}/{selected_volume}'
                files = self.w.files.list_directory_contents(volume_path)
                file_options = [
                    {'label': entry.path.split('/')[-1], 'value': entry.path}
                    for entry in files
                    if entry.path.endswith('.c')
                ]
                
                # Updated default file logic
                default_value = next(
                    (opt['value'] for opt in file_options 
                     if opt['label'] == self.default_file),
                    None
                ) if (
                    selected_catalog == self.default_catalog and 
                    selected_schema == self.default_schema and 
                    selected_volume == self.default_volume
                ) else None
                
                return file_options, default_value
            except Exception as e:
                print(f'Error fetching files: {str(e)}')
                return [], None

        @self.app.callback(
            Output('parse-file-button', 'style'),
            Input('file-picker-dropdown', 'value')
        )
        def toggle_parse_button(selected_file):
            if selected_file and selected_file.endswith('.c'):
                return {'display': 'block'}
            return {'display': 'none'}

        @self.app.callback(
            Output('parse-result', 'children'),
            Input('parse-file-button', 'n_clicks'),
            State('file-picker-dropdown', 'value'),
            prevent_initial_call=True
        )
        def parse_selected_file(n_clicks, file_path):
            if not n_clicks or not file_path:
                return ''
            
            try:
                self.code_analyzer.parse_c_file(file_path)
                return html.Div('File parsed successfully!', 
                              style={'color': 'green', 'marginTop': '10px'})
            except Exception as e:
                # Create a pre-formatted block to preserve whitespace and formatting
                error_message = str(e)
                if "Parse error near line" in error_message:
                    # It's our detailed error message
                    return html.Div([
                        html.Div('Parse error:', 
                                style={'color': 'red', 'marginTop': '10px', 'fontWeight': 'bold'}),
                        html.Pre(error_message,
                                style={'backgroundColor': '#f8f9fa', 
                                      'padding': '10px', 
                                      'borderRadius': '4px',
                                      'whiteSpace': 'pre-wrap',
                                      'fontFamily': 'monospace',
                                      'fontSize': '0.9em'})
                    ])
                else:
                    # It's a different kind of error
                    return html.Div([
                        html.Div('Error:', 
                                style={'color': 'red', 'marginTop': '10px', 'fontWeight': 'bold'}),
                        html.Pre(f"Error processing C file: {str(e)}",
                                style={'backgroundColor': '#f8f9fa', 
                                      'padding': '10px', 
                                      'borderRadius': '4px',
                                      'whiteSpace': 'pre-wrap',
                                      'fontFamily': 'monospace'})
                    ])

        # Add new callback to auto-click parse button when file is selected
        @self.app.callback(
            Output('parse-file-button', 'n_clicks'),
            Input('file-picker-dropdown', 'value'),
            prevent_initial_call=True
        )
        def auto_click_parse(selected_file):
            if selected_file and selected_file.endswith('.c'):
                return 1
            return None

        @self.app.callback(
            Output('symbol-search-container', 'style'),
            Output('symbol-search-dropdown', 'options'),
            Input('parse-result', 'children')
        )
        def update_symbol_search(parse_result):
            if not parse_result or isinstance(parse_result, str):
                return {'display': 'none'}, []
            
            try:
                # Get all variables and functions
                variables = self.code_analyzer.get_all_variables()
                
                # Create options list with types
                options = []
                
                # Add variables
                for var in variables:
                    var_info = self.code_analyzer.get_variable_info(var)
                    if var_info:
                        # Format the label more cleanly
                        function_info = f" in {var_info['function']}" if var_info['function'] else " (global)"
                        label = html.Div([
                            html.Div([
                                html.Span(var, style={'font-weight': 'bold'}),
                                html.Span(function_info, style={'color': '#666', 'font-size': '0.9em'})
                            ]),
                            html.Div(
                                var_info['type'],
                                style={'color': '#0066cc', 'font-size': '0.8em'}
                            )
                        ])
                        
                        options.append({
                            'label': label,
                            'value': f"var:{var}",
                            'search': var  # For better searching
                        })
                
                # Sort options alphabetically by variable name
                options.sort(key=lambda x: x['value'])
                
                return {'display': 'block'}, options
            except Exception as e:
                print(f"Error updating symbol search: {str(e)}")
                return {'display': 'none'}, []

        @self.app.callback(
            Output('symbol-details', 'children'),
            Input('symbol-search-dropdown', 'value')
        )
        def update_symbol_details(selected_value):
            if not selected_value:
                return ''
            
            try:
                # Parse the selection
                symbol_type, symbol_name = selected_value.split(':')
                
                if symbol_type == 'var':
                    # Get variable details
                    var_info = self.code_analyzer.get_variable_info(symbol_name)
                    if var_info:
                        return html.Div([
                            html.H5(f"Variable: {symbol_name}"),
                            html.P([
                                html.Strong("Type: "), var_info['type'],
                                html.Br(),
                                html.Strong("Function: "), var_info['function'] or 'global',
                                html.Br(),
                                html.Strong("Is Pointer: "), str(var_info['is_pointer']),
                                html.Br(),
                                html.Strong("Dependencies: "), 
                                ', '.join(var_info['dependencies']) or 'none',
                                html.Br(),
                                html.Strong("Dependent Variables: "), 
                                ', '.join(var_info['dependents']) or 'none'
                            ])
                        ])
                
                return html.Div("No details available")
                
            except Exception as e:
                print(f"Error showing symbol details: {str(e)}")
                return html.Div(f"Error: {str(e)}")

        @self.app.callback(
            Output('dependency-graph', 'elements'),
            Input('symbol-search-dropdown', 'value')
        )
        def update_dependency_graph(selected_value):
            if not selected_value:
                return []
            
            try:
                # Parse the selection
                symbol_type, symbol_name = selected_value.split(':')
                
                if symbol_type == 'var':
                    # Get the dependency subgraph
                    graph = self.code_analyzer.visualize_dependencies(symbol_name)
                    
                    # Convert networkx graph to cytoscape format
                    elements = []
                    
                    # Add nodes
                    for node in graph.nodes():
                        node_data = {
                            'data': {
                                'id': str(node),
                                'label': str(node),
                                'type': 'constant' if str(node).startswith('Constant:') 
                                        else 'target' if node == symbol_name 
                                        else 'variable'
                            }
                        }
                        elements.append(node_data)
                    
                    # Add edges
                    for source, target in graph.edges():
                        edge_data = {
                            'data': {
                                'source': str(source),
                                'target': str(target)
                            }
                        }
                        elements.append(edge_data)
                    
                    return elements
                
                return []
                
            except Exception as e:
                print(f"Error creating dependency graph: {str(e)}")
                return []
