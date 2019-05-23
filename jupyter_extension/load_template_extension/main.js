// file load_template_extension/main.js

define([
    'require',
    'jquery',
    'base/js/namespace',
    './jszip.min',
    './FileSaver.min'
], function(
    requirejs,
    $,
    Jupyter,
    JSZip,
    aa
) {
    'use strict';
    
    let model_html = 
`
<div class="modal fade" id="pipeChooseModal" tabindex="-1" role="dialog" aria-labelledby="pipeChooseModalLabel">
  <div class="modal-dialog" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
        <h4 class="modal-title" id="pipeChooseModalLabel">Choose Pipeline Template</h4>
      </div>
      <div class="modal-body">
        <div class="row">
          <div class="col-xs-4 col-md-2">
            <a href="#" class="thumbnail">
              <img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9InllcyI/PjxzdmcgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB3aWR0aD0iMTcxIiBoZWlnaHQ9IjE4MCIgdmlld0JveD0iMCAwIDE3MSAxODAiIHByZXNlcnZlQXNwZWN0UmF0aW89Im5vbmUiPjwhLS0KU291cmNlIFVSTDogaG9sZGVyLmpzLzEwMCV4MTgwCkNyZWF0ZWQgd2l0aCBIb2xkZXIuanMgMi42LjAuCkxlYXJuIG1vcmUgYXQgaHR0cDovL2hvbGRlcmpzLmNvbQooYykgMjAxMi0yMDE1IEl2YW4gTWFsb3BpbnNreSAtIGh0dHA6Ly9pbXNreS5jbwotLT48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwhW0NEQVRBWyNob2xkZXJfMTY5YWVjMjkzYmQgdGV4dCB7IGZpbGw6I0FBQUFBQTtmb250LXdlaWdodDpib2xkO2ZvbnQtZmFtaWx5OkFyaWFsLCBIZWx2ZXRpY2EsIE9wZW4gU2Fucywgc2Fucy1zZXJpZiwgbW9ub3NwYWNlO2ZvbnQtc2l6ZToxMHB0IH0gXV0+PC9zdHlsZT48L2RlZnM+PGcgaWQ9ImhvbGRlcl8xNjlhZWMyOTNiZCI+PHJlY3Qgd2lkdGg9IjE3MSIgaGVpZ2h0PSIxODAiIGZpbGw9IiNFRUVFRUUiLz48Zz48dGV4dCB4PSI1OS41NjI1IiB5PSI5NC41NTYyNSI+MTcxeDE4MDwvdGV4dD48L2c+PC9nPjwvc3ZnPg==" alt="...">
              <div class="caption" style="padding: 2px 9px;">
                <h6 class="text-center">Template 1</h6>
                <p>...</p>
              </div>
            </a>
          </div>
          <div class="col-xs-4 col-md-2">
            <a href="#" class="thumbnail">
              <img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9InllcyI/PjxzdmcgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB3aWR0aD0iMTcxIiBoZWlnaHQ9IjE4MCIgdmlld0JveD0iMCAwIDE3MSAxODAiIHByZXNlcnZlQXNwZWN0UmF0aW89Im5vbmUiPjwhLS0KU291cmNlIFVSTDogaG9sZGVyLmpzLzEwMCV4MTgwCkNyZWF0ZWQgd2l0aCBIb2xkZXIuanMgMi42LjAuCkxlYXJuIG1vcmUgYXQgaHR0cDovL2hvbGRlcmpzLmNvbQooYykgMjAxMi0yMDE1IEl2YW4gTWFsb3BpbnNreSAtIGh0dHA6Ly9pbXNreS5jbwotLT48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwhW0NEQVRBWyNob2xkZXJfMTY5YWVjMjkzYmQgdGV4dCB7IGZpbGw6I0FBQUFBQTtmb250LXdlaWdodDpib2xkO2ZvbnQtZmFtaWx5OkFyaWFsLCBIZWx2ZXRpY2EsIE9wZW4gU2Fucywgc2Fucy1zZXJpZiwgbW9ub3NwYWNlO2ZvbnQtc2l6ZToxMHB0IH0gXV0+PC9zdHlsZT48L2RlZnM+PGcgaWQ9ImhvbGRlcl8xNjlhZWMyOTNiZCI+PHJlY3Qgd2lkdGg9IjE3MSIgaGVpZ2h0PSIxODAiIGZpbGw9IiNFRUVFRUUiLz48Zz48dGV4dCB4PSI1OS41NjI1IiB5PSI5NC41NTYyNSI+MTcxeDE4MDwvdGV4dD48L2c+PC9nPjwvc3ZnPg==" alt="...">
              <div class="caption" style="padding: 2px 9px;">
                <h6 class="text-center">Template 2</h6>
                <p>...</p>
              </div>
            </a>
          </div>
          <div class="col-xs-4 col-md-2">
            <a href="#" class="thumbnail">
              <img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9InllcyI/PjxzdmcgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB3aWR0aD0iMTcxIiBoZWlnaHQ9IjE4MCIgdmlld0JveD0iMCAwIDE3MSAxODAiIHByZXNlcnZlQXNwZWN0UmF0aW89Im5vbmUiPjwhLS0KU291cmNlIFVSTDogaG9sZGVyLmpzLzEwMCV4MTgwCkNyZWF0ZWQgd2l0aCBIb2xkZXIuanMgMi42LjAuCkxlYXJuIG1vcmUgYXQgaHR0cDovL2hvbGRlcmpzLmNvbQooYykgMjAxMi0yMDE1IEl2YW4gTWFsb3BpbnNreSAtIGh0dHA6Ly9pbXNreS5jbwotLT48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwhW0NEQVRBWyNob2xkZXJfMTY5YWVjMjkzYmQgdGV4dCB7IGZpbGw6I0FBQUFBQTtmb250LXdlaWdodDpib2xkO2ZvbnQtZmFtaWx5OkFyaWFsLCBIZWx2ZXRpY2EsIE9wZW4gU2Fucywgc2Fucy1zZXJpZiwgbW9ub3NwYWNlO2ZvbnQtc2l6ZToxMHB0IH0gXV0+PC9zdHlsZT48L2RlZnM+PGcgaWQ9ImhvbGRlcl8xNjlhZWMyOTNiZCI+PHJlY3Qgd2lkdGg9IjE3MSIgaGVpZ2h0PSIxODAiIGZpbGw9IiNFRUVFRUUiLz48Zz48dGV4dCB4PSI1OS41NjI1IiB5PSI5NC41NTYyNSI+MTcxeDE4MDwvdGV4dD48L2c+PC9nPjwvc3ZnPg==" alt="...">
              <div class="caption" style="padding: 2px 9px;">
                <h6 class="text-center">Template 3</h6>
                <p>...</p>
              </div>
            </a>
          </div>
          <div class="col-xs-4 col-md-2">
            <a href="#" class="thumbnail">
              <img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9InllcyI/PjxzdmcgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB3aWR0aD0iMTcxIiBoZWlnaHQ9IjE4MCIgdmlld0JveD0iMCAwIDE3MSAxODAiIHByZXNlcnZlQXNwZWN0UmF0aW89Im5vbmUiPjwhLS0KU291cmNlIFVSTDogaG9sZGVyLmpzLzEwMCV4MTgwCkNyZWF0ZWQgd2l0aCBIb2xkZXIuanMgMi42LjAuCkxlYXJuIG1vcmUgYXQgaHR0cDovL2hvbGRlcmpzLmNvbQooYykgMjAxMi0yMDE1IEl2YW4gTWFsb3BpbnNreSAtIGh0dHA6Ly9pbXNreS5jbwotLT48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwhW0NEQVRBWyNob2xkZXJfMTY5YWVjMjkzYmQgdGV4dCB7IGZpbGw6I0FBQUFBQTtmb250LXdlaWdodDpib2xkO2ZvbnQtZmFtaWx5OkFyaWFsLCBIZWx2ZXRpY2EsIE9wZW4gU2Fucywgc2Fucy1zZXJpZiwgbW9ub3NwYWNlO2ZvbnQtc2l6ZToxMHB0IH0gXV0+PC9zdHlsZT48L2RlZnM+PGcgaWQ9ImhvbGRlcl8xNjlhZWMyOTNiZCI+PHJlY3Qgd2lkdGg9IjE3MSIgaGVpZ2h0PSIxODAiIGZpbGw9IiNFRUVFRUUiLz48Zz48dGV4dCB4PSI1OS41NjI1IiB5PSI5NC41NTYyNSI+MTcxeDE4MDwvdGV4dD48L2c+PC9nPjwvc3ZnPg==" alt="...">
              <div class="caption" style="padding: 2px 9px;">
                <h6 class="text-center">Template 4</h6>
                <p>...</p>
              </div>
            </a>
          </div>
          <div class="col-xs-4 col-md-2">
            <a href="#" class="thumbnail">
              <img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9InllcyI/PjxzdmcgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB3aWR0aD0iMTcxIiBoZWlnaHQ9IjE4MCIgdmlld0JveD0iMCAwIDE3MSAxODAiIHByZXNlcnZlQXNwZWN0UmF0aW89Im5vbmUiPjwhLS0KU291cmNlIFVSTDogaG9sZGVyLmpzLzEwMCV4MTgwCkNyZWF0ZWQgd2l0aCBIb2xkZXIuanMgMi42LjAuCkxlYXJuIG1vcmUgYXQgaHR0cDovL2hvbGRlcmpzLmNvbQooYykgMjAxMi0yMDE1IEl2YW4gTWFsb3BpbnNreSAtIGh0dHA6Ly9pbXNreS5jbwotLT48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwhW0NEQVRBWyNob2xkZXJfMTY5YWVjMjkzYmQgdGV4dCB7IGZpbGw6I0FBQUFBQTtmb250LXdlaWdodDpib2xkO2ZvbnQtZmFtaWx5OkFyaWFsLCBIZWx2ZXRpY2EsIE9wZW4gU2Fucywgc2Fucy1zZXJpZiwgbW9ub3NwYWNlO2ZvbnQtc2l6ZToxMHB0IH0gXV0+PC9zdHlsZT48L2RlZnM+PGcgaWQ9ImhvbGRlcl8xNjlhZWMyOTNiZCI+PHJlY3Qgd2lkdGg9IjE3MSIgaGVpZ2h0PSIxODAiIGZpbGw9IiNFRUVFRUUiLz48Zz48dGV4dCB4PSI1OS41NjI1IiB5PSI5NC41NTYyNSI+MTcxeDE4MDwvdGV4dD48L2c+PC9nPjwvc3ZnPg==" alt="...">
              <div class="caption" style="padding: 2px 9px;">
                <h6 class="text-center">Template 5</h6>
                <p>...</p>
              </div>
            </a>
          </div>
          <div class="col-xs-4 col-md-2">
            <a href="#" class="thumbnail">
              <img src="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9InllcyI/PjxzdmcgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB3aWR0aD0iMTcxIiBoZWlnaHQ9IjE4MCIgdmlld0JveD0iMCAwIDE3MSAxODAiIHByZXNlcnZlQXNwZWN0UmF0aW89Im5vbmUiPjwhLS0KU291cmNlIFVSTDogaG9sZGVyLmpzLzEwMCV4MTgwCkNyZWF0ZWQgd2l0aCBIb2xkZXIuanMgMi42LjAuCkxlYXJuIG1vcmUgYXQgaHR0cDovL2hvbGRlcmpzLmNvbQooYykgMjAxMi0yMDE1IEl2YW4gTWFsb3BpbnNreSAtIGh0dHA6Ly9pbXNreS5jbwotLT48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwhW0NEQVRBWyNob2xkZXJfMTY5YWVjMjkzYmQgdGV4dCB7IGZpbGw6I0FBQUFBQTtmb250LXdlaWdodDpib2xkO2ZvbnQtZmFtaWx5OkFyaWFsLCBIZWx2ZXRpY2EsIE9wZW4gU2Fucywgc2Fucy1zZXJpZiwgbW9ub3NwYWNlO2ZvbnQtc2l6ZToxMHB0IH0gXV0+PC9zdHlsZT48L2RlZnM+PGcgaWQ9ImhvbGRlcl8xNjlhZWMyOTNiZCI+PHJlY3Qgd2lkdGg9IjE3MSIgaGVpZ2h0PSIxODAiIGZpbGw9IiNFRUVFRUUiLz48Zz48dGV4dCB4PSI1OS41NjI1IiB5PSI5NC41NTYyNSI+MTcxeDE4MDwvdGV4dD48L2c+PC9nPjwvc3ZnPg==" alt="...">
              <div class="caption" style="padding: 2px 9px;">
                <h6 class="text-center">Template 6</h6>
                <p>...</p>
              </div>
            </a>
          </div>
        </div>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-sm btn-default" data-dismiss="modal">Close</button>
        <button type="button" class="btn btn-sm btn-primary" id="pipeChooseBtn">Load</button>
      </div>
    </div>
  </div>
</div>
`;
    
    function load_ipython_extension() {
        
        let load_code_from_py = function (url) {
            
            $.get(url, function(result) {               
                Jupyter.notebook.insert_cell_at_bottom();
                let last_cell = Jupyter.notebook.get_cell(-1);
                last_cell.set_text(result);
            });
            
        };
        
        let load_cells_from_notebook_old = function (url) {
            
            $.getJSON(url, function(result) {
                let cells = result.cells;
                for (let i=0; i<cells.length; i++) {
                    Jupyter.notebook.insert_cell_at_bottom();
                    let last_cell = Jupyter.notebook.get_cell(-1);
                    cells[i]['source'] = cells[i]['source'].join('');
                    last_cell.fromJSON(cells[i]);
                };
            });
            
        };
        
        let load_cells_from_notebook = function (url) {
            
            $.getJSON(url, function(result) {             
                result['cells'].forEach((cell) => {
                    cell['source'] = cell['source'].join('');
                    Jupyter.notebook.insert_cell_at_bottom();
                });
                let json = {
                    'name'    : Jupyter.notebook.notebook_name,
                    'path'    : Jupyter.notebook.notebook_path,
                    'content' : result
                }
                Jupyter.notebook.fromJSON(json);
            });
            
        };
        
        $('body').append(model_html);
        
        $('#pipeChooseBtn').click(function() {
            
            $('#pipeChooseModal').modal('hide');
            
            // Load cells from .ipynb (notebook level)
            let notebook_url = 'http://localhost:8888/files/GUI/ipyantd_test.ipynb?download=1';
            load_cells_from_notebook(notebook_url);
            
        });
        
        let action1 = {
            icon       : 'fa-bolt',
            help       : 'Run Jupyter macro',
            help_index : 'zz',
            handler    : () => {
                $('#pipeChooseModal').modal('show');
            }
        };
        let prefix1 = 'load_template_extension';
        let action_name1 = 'show-create-gui1';
        let full_action_name1 = Jupyter.actions.register(action1, action_name1, prefix1);
        
        let action2 = {
            icon       : 'fa-star',
            help       : 'Run Jupyter macro',
            help_index : 'zz',
            handler    : () => {
                let notebook_json = Jupyter.notebook.toJSON();
                let py_code = "";
                notebook_json['cells'].forEach((cell) => {
                    if (cell.cell_type == 'code') {
                        console.log(cell.source);
                        py_code += cell.source + "\n";
                    }
                });
                var zip = new JSZip();
                zip.file("main.py", py_code);
                //var img = zip.folder("images");
                //img.file("smile.gif", imgData, {base64: true});
                zip.generateAsync({type:"blob"})
                .then(function(content) {
                    //aa.saveAs(content, "example.zip");
                    location.href = window.URL.createObjectURL(content);
                });
            }
        };
        let prefix2 = 'load_template_extension';
        let action_name2 = 'show-create-gui2';
        let full_action_name2 = Jupyter.actions.register(action2, action_name2, prefix2);       
        
        Jupyter.toolbar.add_buttons_group([full_action_name1, full_action_name2]);
    }

    return {
        load_ipython_extension: load_ipython_extension
    };
    
});
