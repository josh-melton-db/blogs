from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from ..services.code_analyzer import CodeAnalyzer

class ChatInterface:
    def __init__(self, app, workspace_client, endpoint_name):
        self.app = app
        self.w = workspace_client
        self.endpoint_name = endpoint_name
        self.code_analyzer = CodeAnalyzer(workspace_client)
        self._create_callbacks()

    def _create_callbacks(self):
        # New callback for immediate message display
        @self.app.callback(
            Output('chat-history-store', 'data', allow_duplicate=True),
            Output('user-input', 'value', allow_duplicate=True),
            Input('send-button', 'n_clicks'),
            Input('user-input', 'n_submit'),
            State('user-input', 'value'),
            State('chat-history-store', 'data'),
            prevent_initial_call=True
        )
        def add_user_message(send_clicks, n_submit, user_input, chat_history):
            if not user_input:
                return chat_history or [], ''
            
            if not chat_history:
                chat_history = []
                
            chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            return chat_history, ''

        # Modified callback for AI response
        @self.app.callback(
            Output('chat-history-store', 'data', allow_duplicate=True),
            Input('chat-history-store', 'data'),
            State('file-picker-dropdown', 'value'),
            State('symbol-search-dropdown', 'value'),
            prevent_initial_call=True
        )
        def get_ai_response(chat_history, file_path, selected_symbol):
            if not chat_history or chat_history[-1]['role'] != 'user':
                return chat_history or []

            try:
                # Get the last user message
                last_message = chat_history[-1]['content']
                
                # Prepare focused context from the selected file and symbol
                code_context = ""
                if file_path and selected_symbol:
                    try:
                        # Parse the symbol type and name
                        symbol_type, symbol_name = selected_symbol.split(':')
                        
                        if symbol_type == 'var':
                            # Get variable details using the existing method
                            var_info = self.code_analyzer.get_variable_info(symbol_name)
                            if var_info:
                                code_context = (
                                    f"\nContext:\n"
                                    f"- File: {file_path}\n"
                                    f"- Variable: {symbol_name}\n"
                                    f"- Type: {var_info['type']}\n"
                                    f"- Scope: {var_info['function'] or 'global'}\n"
                                    f"- Is Pointer: {var_info['is_pointer']}\n"
                                    f"- Dependencies: {', '.join(var_info['dependencies']) or 'none'}\n"
                                    f"- Used By: {', '.join(var_info['dependents']) or 'none'}\n"
                                )
                    except Exception as e:
                        code_context = f"\nNote: Failed to get variable details: {str(e)}\n"
                
                # Create a separate messages list for the API call
                api_messages = []
                
                # Add context as a system message if it exists
                if code_context:
                    api_messages.append(ChatMessage(
                        content=code_context,
                        role=ChatMessageRole.SYSTEM
                    ))
                
                # Add all user and assistant messages from chat history
                for msg in chat_history:
                    api_messages.append(ChatMessage(
                        content=msg['content'],
                        role=ChatMessageRole[msg['role'].upper()]
                    ))
                
                # Get AI response
                response = self.w.serving_endpoints.query(
                    name=self.endpoint_name,
                    messages=api_messages,
                )
                
                # Add AI response to visible chat history
                chat_history.append({
                    'role': 'assistant',
                    'content': response.choices[0].message.content
                })
                
            except Exception as e:
                print(f'Error getting response: {str(e)}')
                chat_history.append({
                    'role': 'assistant',
                    'content': f'Error: {str(e)}'
                })
            
            return chat_history

        @self.app.callback(
            Output('chat-history', 'children'),
            Input('chat-history-store', 'data')
        )
        def update_chat_display(chat_history):
            if not chat_history:
                return []
            
            chat_elements = []
            for message in chat_history:
                role = message['role']
                content = message['content']
                
                style = {
                    'padding': '10px',
                    'margin': '5px',
                    'border-radius': '10px',
                    'max-width': '70%',
                }
                
                if role == 'user':
                    style.update({
                        'background-color': '#DCF8C6',
                        'margin-left': 'auto',
                    })
                else:
                    style.update({
                        'background-color': '#E8E8E8',
                        'margin-right': 'auto',
                    })
                
                chat_elements.append(
                    html.Div(content, style=style)
                )
            
            return chat_elements

        @self.app.callback(
            Output('symbol-search-dropdown', 'value', allow_duplicate=True),
            Input('dependency-graph', 'tapNodeData'),
            prevent_initial_call=True
        )
        def handle_node_click(node_data):
            if node_data:
                node_id = node_data.get('id')
                if node_id:
                    # Add the 'var:' prefix to match the symbol search format
                    return f'var:{node_id}'
            return None

    def create_chat_layout(self):
        return html.Div([
            dcc.Store(id='chat-history-store', data=[]),
            html.Div(
                id='chat-history', 
                style={
                    'height': '200px',  # Reduced from 400px
                    'overflow-y': 'auto',
                    'border': '1px solid #ddd',
                    'border-radius': '4px',
                    'margin-bottom': '10px'
                }
            ),
            dbc.Row([
                dbc.Col([
                    dbc.Input(
                        id='user-input',
                        type='text',
                        placeholder='Type your message...',
                        style={'width': '100%'}
                    ),
                ], width=10),
                dbc.Col([
                    dbc.Button('Send', id='send-button', color='primary', className='me-2'),
                    dbc.Button('Clear', id='clear-button', color='secondary')
                ], width=2)
            ], className='mt-2')
        ], className='bg-white p-3 rounded shadow-sm')

