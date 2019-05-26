// file load_template_extension/main.js

define([
    'require',
    'jquery',
    'base/js/namespace',
    './jszip.min',
    './jszip-utils.min',
    './FileSaver.min'
], function(
    requirejs,
    $,
    Jupyter,
    JSZip,
    JSZipUtils,
    FileSaver
) {
    'use strict';
    
    let model1_html = 
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
              <img src="" alt="...">
              <div class="caption" style="padding: 2px 9px;">
                <h6 class="text-center">Template 1</h6>
                <p>...</p>
              </div>
            </a>
          </div>
          <div class="col-xs-4 col-md-2">
            <a href="#" class="thumbnail">
              <img src="" alt="...">
              <div class="caption" style="padding: 2px 9px;">
                <h6 class="text-center">Template 2</h6>
                <p>...</p>
              </div>
            </a>
          </div>
          <div class="col-xs-4 col-md-2">
            <a href="#" class="thumbnail">
              <img src="" alt="...">
              <div class="caption" style="padding: 2px 9px;">
                <h6 class="text-center">Template 3</h6>
                <p>...</p>
              </div>
            </a>
          </div>
          <div class="col-xs-4 col-md-2">
            <a href="#" class="thumbnail">
              <img src="" alt="...">
              <div class="caption" style="padding: 2px 9px;">
                <h6 class="text-center">Template 4</h6>
                <p>...</p>
              </div>
            </a>
          </div>
          <div class="col-xs-4 col-md-2">
            <a href="#" class="thumbnail">
              <img src="" alt="...">
              <div class="caption" style="padding: 2px 9px;">
                <h6 class="text-center">Template 5</h6>
                <p>...</p>
              </div>
            </a>
          </div>
          <div class="col-xs-4 col-md-2">
            <a href="#" class="thumbnail">
              <img src="" alt="...">
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

    let model2_html = 
`
<div class="modal fade" id="pkgModal" tabindex="-1" role="dialog" aria-labelledby="pkgModalLabel">
  <div class="modal-dialog" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
        <h4 class="modal-title" id="pkgModalLabel">Automatic packaging...</h4>
      </div>
      <div class="modal-body">
        <div class="row">
          <div class="col-xs-12 col-md-12">
            <div class="progress">
              <div id='pkg-progressbar' class="progress-bar" role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%;">
                0%
              </div>
            </div>
            <p id='pkg-text'>Initial...</p>
          </div>
        </div>
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
        
        $('body').append(model1_html);
        $('body').append(model2_html);
        
        $('#pipeChooseBtn').click(function() {
            
            $('#pipeChooseModal').modal('hide');
            
            // Load cells from .ipynb (notebook level)
            let notebook_url = 'http://localhost:8888/files/GUI/ipyantd_test.ipynb?download=1';
            load_cells_from_notebook(notebook_url);
            
        });
        
        let action1 = {
            icon       : 'fa-bolt',
            help       : 'Load template',
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
            help       : 'Automatic packaging',
            help_index : 'zz',
            handler    : () => {
                
                $('#pkgModal').modal('show');
                
                let $pkg_progs_bar = $('#pkg-progressbar');
                let $pkg_text = $('#pkg-text');
                
                let source_zip_url = 'http://localhost:8888/files/CustomPkgs/main.zip?download=1';
                
                let notebook_json = Jupyter.notebook.toJSON();
                let py_code = "def run(**args):\n";
                notebook_json['cells'].forEach((cell) => {
                    if (cell.cell_type == 'code') {
                        console.log(cell.source);
                        py_code += "    " + cell.source.replace(/(?:\r\n|\r|\n)/g, "\n    ") + "\n\n";
                    }
                });
                
                JSZipUtils.getBinaryContent(source_zip_url, function(err, data) {
                    if (err) {
                        throw err;
                    }

                    JSZip.loadAsync(data).then(function (zip) {
                        zip.file("main.py", py_code);
                        zip.generateAsync({
                            type: "blob",
                            compression: "DEFLATE",
                            compressionOptions: {
                                level: 6
                            }
                        }, function (metadata) {
                            $pkg_progs_bar.width( metadata.percent.toFixed(2) + "%" ).text( metadata.percent.toFixed(2) + "%" );
                            if (metadata.currentFile) {
                                $pkg_text.text("File: " + metadata.currentFile);
                            }
                        }).then(function(content) {
                            $('#pkgModal').modal('hide');
                            saveAs(content, "main.zip");
                        });
                    });
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
