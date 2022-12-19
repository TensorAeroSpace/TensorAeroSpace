/* Copyright 2013-2015 The MathWorks, Inc. */
function queryByClassName(className, elem) {
    if (!elem) elem = document.body;
    if (typeof elem.querySelectorAll === "function") {
        return elem.querySelectorAll("."+className);
    } else {
        return elem.getElementsByClass(className);
    }
}

function nav_token_usage_details(direction) {
    var els = queryByClassName("token_usage_details_tabrow");
    var selectedIdx = 0;
    var selectedClassName = "selected";
    for (selectedIdx; selectedIdx < els.length; selectedIdx++) {
        if (els[selectedIdx].classList.contains(selectedClassName)) {
            break;
        }
    }
    var nextIdx = selectedIdx;
    if (direction === -1 && selectedIdx > 0) {
        nextIdx = selectedIdx-1;                
    } else if (direction === 1 && selectedIdx < els.length - 1) {
        nextIdx = selectedIdx + 1;
    }   
    if (nextIdx !== selectedIdx) {
        els[selectedIdx].classList.remove(selectedClassName);
        els[nextIdx].classList.add(selectedClassName);
        els[nextIdx].scrollIntoView(alignWithTop=false);
    }
    return false;
}

function tabrowClicked(event) { 
}

function popupOnload() {
    var els = queryByClassName("token_usage_details_tabrow");   
    for (var i=0; i<els.length; i++) {
        els[i].onclick= tabrowClicked;
    }
};

function tokenOnRightclick(event) {
    var filename = location.pathname.split(/\//);
    filename = filename[filename.length-1];
    top.inspectToken(filename, location.pathname, event);
    top.hiliteClickedToken(event.currentTarget);
    return false;
}

function tokenOnclick(event) {
    tokenOnRightclick(event);
    if (event.currentTarget.href.length !== 0 && event.currentTarget.href.protocol !== "matlab:") {
        top.tokenLinkOnClick(event);
        return true;
    }
    return false;
};

function tokenOnMouseOver(event) {    
    var filename = location.pathname.split(/\//);
    filename = filename[filename.length-1];
    createPopup(filename, event);
};
function tokenOnMouseOut(event) {
    destroyPopup(event.currentTarget);
};

function blkLinkOnClick(event) {
    top.hiliteClickedToken(event.currentTarget);
    return true;
}
function clearTokenLink(id) {    
    var codeElement = document.getElementById(id);
    var els = queryByClassName("tk", codeElement);
    var elem; 
    if (top.CodeDefine && top.CodeDefine.instance)  {
        for (var i=0; i<els.length; i++) {
            var re = new RegExp('active', 'g');
            els[i].className = els[i].className.replace(re, '');
            re = new RegExp('traceable_token', 'g');
            els[i].className = els[i].className.replace(re, '');            
        }
    }
}
function updateTokenLink(id) {
    var codeElement = document.getElementById(id);
    var filename = location.pathname.split(/\//);
    filename = filename[filename.length-1];  
    var srcFilename;
    if (top.RTW_TraceInfo) {
        srcFilename = top.RTW_TraceInfo.toSrcFileName(filename);
    }
    var els = queryByClassName("tk", codeElement);
    var elem; 
    var hasTraceFlag = null;
    if (top.TraceInfoFlag && top.TraceInfoFlag.instance)
        hasTraceFlag =  top.TraceInfoFlag.instance.traceFlag;
    else 
        hasTraceFlag = false;
    var defObj;
    var traceObj;
    var codeDef = (top.CodeDefine &&  top.CodeDefine.instance) ? top.CodeDefine.instance : null;
    var aLink = top.document.createElement("a");
    if (hasTraceFlag || (top.CodeDefine && top.CodeDefine.instance))  {
        for (var i=0; i<els.length; i++) {
            defObj = (codeDef) ? codeDef.def[els[i].text] : null;
            traceObj = hasTraceFlag && hasTraceFlag[srcFilename+":"+els[i].id];
            if (traceObj || defObj) {
                els[i].onclick= tokenOnclick;
                els[i].oncontextmenu= tokenOnRightclick;
                els[i].onmouseover = tokenOnMouseOver;    
                els[i].onmouseout = tokenOnMouseOut;
                els[i].className += " active";
                els[i].target = "rtwreport_document_frame";
                if (traceObj && top.reportModel) {    
                    if (top.testHarnessInfo && top.testHarnessInfo.IsTestHarness==="1") {
                        els[i].href = "matlab:rtw.report.code2model('" + top.reportModel 
                            + "','" + location.pathname + "','" + els[i].id 
                            + "','" + top.testHarnessInfo.HarnessName 
                            + "','" + top.testHarnessInfo.HarnessOwner
                            + "','" + top.testHarnessInfo.OwnerFileName + "')";
                    } else {
                        els[i].href = "matlab:rtw.report.code2model(\'" + top.reportModel 
                            + "\',\'" + location.pathname + "\',\'" + els[i].id + "')";
                    }
                } else {
                    aLink.href = defObj.file;
                    els[i].href = aLink.pathname + "#" + defObj.line;
                }
            }
        }
    }
}
// remove the code table; insert back
function updateToken(codeElement) {
    var filename = location.pathname.split(/\//);
    filename = filename[filename.length-1];  
    var srcFilename;
    if (top.RTW_TraceInfo) {
        srcFilename = top.RTW_TraceInfo.toSrcFileName(filename);
    }
    // update block path link in comments
    els = queryByClassName("blk", codeElement);
    var lineSid = null;
    if (top.RTW_rtwnameSIDMap && top.RTW_rtwnameSIDMap.instance) {
        for (var i=0; i<els.length; i++) {
            lineSid = top.RTW_rtwnameSIDMap.instance.getSID(els[i].text);
            if (lineSid) {
                if (top.testHarnessInfo && top.testHarnessInfo.IsTestHarness==="1") {
                    els[i].href = "matlab:coder.internal.code2model('" + lineSid.sid + "','" +
                        top.testHarnessInfo.HarnessName+ "','" +
                        top.testHarnessInfo.HarnessOwner+ "','" + 
                        top.testHarnessInfo.OwnerFileName + "');";
                } else {
                    els[i].href = "matlab:coder.internal.code2model('" + lineSid.sid + "');";
                }
                els[i].id = "blkLink_" + i;
                els[i].onclick = blkLinkOnClick;
                els[i].className += " blk_active";
            }
        }
    }   
    // update lib block path link in comments
    els = queryByClassName("libblk", codeElement);
    var lineSid = null;    
    for (var i=0; i<els.length; i++) {
        lineSid = els[i].text;
        if (lineSid) {
            if (top.testHarnessInfo && top.testHarnessInfo.IsTestHarness==="1") {
                els[i].href = "matlab:coder.internal.code2model('" + lineSid + "','" +
                    top.testHarnessInfo.HarnessName+ "','" +
                    top.testHarnessInfo.HarnessOwner+ "','" + 
                    top.testHarnessInfo.OwnerFileName + "');";
            } else {
                els[i].href = "matlab:coder.internal.code2model('" + lineSid + "');";
            }
            els[i].id = "blkLink_" + i;
            els[i].onclick = blkLinkOnClick;
            els[i].className += " blk_active";
        }
    }
    // update requirement link in comments
    els = queryByClassName("req", codeElement);
    var req_block;
    if (top.RTW_rtwnameSIDMap && top.RTW_rtwnameSIDMap.instance) {
        for (var i=0; i<els.length; i++) {
            lineSid = top.RTW_rtwnameSIDMap.instance.getSID(els[i].getAttribute("blockpath"));
            if (lineSid) {
                req_block = lineSid.sid;
            } else {
                req_block = els[i].getAttribute("blockpath");
            }
            var req_id = els[i].getAttribute("req_id");
            els[i].href = "matlab:rtw.report.code2req('" + req_block + "'," + req_id + ");";
            els[i].id = "req_" + i;
            els[i].onclick = top.reqOnClick;
            els[i].className += " req_active";
        }
    }
    
    // add link to source file
    if (top.Html2SrcLink && top.Html2SrcLink.instance) {
        filename = top.rtwGetFileName(location.href);
        var link2Src = top.Html2SrcLink.instance.getLink2Src(filename);
        var link = document.createElement("h4");
        link.innerHTML = "File: <a href=\"" + link2Src + 
            "\" target = \"rtwreport_document_frame\" id=\"linkToText_plain\">" + 
            top.rtwGetFileName(link2Src) + "</a>";
        var bodyNode = document.body;
        bodyNode.insertBefore(link, bodyNode.firstElementChild);
    }
    top.updateHyperlinks();
    // update fileSelector frame
    if (top.fileSelector) {
        var o = top.fileSelector.document.getElementById('fileSelector');
        if (o) {
            o.value = filename;
        }
    }
}
function getInsertFunction(element) {
    var parentNode = element.parentNode;
    var nextSibling = element.nextSibling;
    parentNode.removeChild(element);
    var spinner = document.createElement("img");
    spinner.src = "spinner.gif";
    parentNode.appendChild(spinner);
    return function() {
        if (spinner) {
            parentNode.removeChild(spinner);
        }
        if (nextSibling) {
            parentNode.insertBefore(element, nextSibling);
        } else {
            parentNode.appendChild(element);
        }
    };
}

var hovered_line = '';
var lineOnMouseIn = function (id) {
    if (hovered_line !== id) {
        hovered_line = id;      
        updateTokenLink(id);            
    }
}
var lineOnMouseOut = function (id) {    
    clearTokenLink(id);
    hovered_line = '';
}
function registerDelayedOnMouseOver(elm, onMouseIn, onMouseOut) {
    var delay = function (elem, onMouseIn, onMouseOut) {                
        var timeout = null;
        elem.onmouseover = function(e) {                                
            timeout = setTimeout(onMouseIn, 200, e.currentTarget.id);
        };
        elem.onmouseout = function(e) {                 
            clearTimeout(timeout);
            if (hovered_line !== '') {
                onMouseOut(e.currentTarget.id);
            }
        }               
    };
    delay(elm, onMouseIn, onMouseOut);
}
// the onload function for source file
function srcFileOnload() {
    var codeElement = document.getElementById("codeTbl");
    var insertFunction = getInsertFunction(codeElement);                
    try {
        var els = codeElement.getElementsByTagName("tr");
        for (var i = 0; i < els.length; i++) {                          
            registerDelayedOnMouseOver(els[i], lineOnMouseIn, lineOnMouseOut);
        }
        updateToken(codeElement);
    } catch (err) {};
    insertFunction();
    // add code to model hyperlinks for all tokens
    var filename = location.pathname.split(/\//);
    filename = filename[filename.length-1];
    // highlight the filename in the TOC frame
    if (top.rtwreport_contents_frame && top.hiliteByFileName(top.rtwreport_document_frame.document.location.href)) {
        // remove the highlights in the TOC frame if filename is hilite successfully
        top.removeHiliteTOC(top.rtwreport_contents_frame);
    }

    // annotate code with code coverage data
    if (typeof rtwannotate === 'function') {
        rtwannotate(filename.replace(/.html$/,"_cov.xml"));
    }

    // highlight token and row 
    if (top.RTW_TraceInfo.instance && top.RTW_TraceArgs.instance) {
        var i;
        // find the highlight file name
        var fileIdx = top.RTW_TraceArgs.instance.getFileIdx(filename);
        var ids=[], rows=[];
        if (typeof fileIdx !== "undefined") {
            ids = top.RTW_TraceArgs.instance.getIDs(fileIdx);
            rows = top.RTW_TraceArgs.instance.getRows(fileIdx);
            // highlight rows in file
            for (i=0; i<rows.length;i++) {
                elem = top.rtwreport_document_frame.document.getElementById(rows[i]);
                if (elem) elem.className += " hilite";
            }
            // highlight tokens in file
            if (top.GlobalConfig.hiliteToken) {
                for (i=0; i<ids.length;i++) {
                    elem = top.rtwreport_document_frame.document.getElementById(ids[i]);
                    if (elem) elem.className += " hilite";              
                }
            }
        } // end of if current file has highlighted lines

        // if the loaded file is not currFile, call setInitLocation
        var currFileIdx = top.RTW_TraceInfo.instance.getCurrFileIdx();
        var myFileIdx = top.RTW_TraceInfo.instance.getFileIdx(filename);
        // update navigation status if the file is loaded first time
        if (currFileIdx !== myFileIdx && document.location.hash === "") {
            if (rows.length > 0) 
                top.RTW_TraceInfo.instance.setInitLocation(filename,rows[0]);
            else {
                top.toggleNavSideBar("off");                            
                return;
            }
        }

        // display navigation side bar
        if (top.rtwreport_nav_frame) top.rtwreport_nav_frame.location.reload();
        if (rows.length>0) {
            top.toggleNavSideBar("on");
            top.toggleNavToolBar("on");
        } else {
            top.toggleNavSideBar("off");
        }
    }
    top.scrollToLineBasedOnHash(document.location.hash);
    function getHash() {
        var loc;
        var aHash="";
        var topDocObj = top.window.document;    
        // get the hash value from location.
        loc = topDocObj.location;
        loc = loc.search || loc.hash;
        aHash = loc.substring(1);
        aHash = decodeURI(aHash);
        return aHash;   
    }
}

function createPopup(filename, evt) {
    var anchorObj = evt.currentTarget;
    if (anchorObj.children.length > 0)
        return;
    var filename = location.pathname.split(/\//);
    filename = filename[filename.length-1];
    var windowObj = top.getInspectWindow();
    var propObj = top.getInspectData(filename, anchorObj);
    var navObj = top.getInspectLink(filename, location.pathname, anchorObj);
    if (propObj) { 
        windowObj.appendChild(propObj);
        windowObj.style.left = "0px";
        if (anchorObj.parentElement.nodeName === "TD" && 
            anchorObj.parentElement.parentElement.nodeName === "TR") {
            anchorObj.parentElement.parentElement.lastChild.insertBefore(windowObj,
                                                                         anchorObj.parentElement.parentElement.lastChild.lastChild.nextSibling);
            var left = Math.min(evt.clientX , window.innerWidth - windowObj.scrollWidth - 30);
            left = Math.max(0, left);
            windowObj.style.left = "" + left + "px";
        }
    }
};

function destroyPopup(anchorObj) {
    var popWindow = document.getElementById("popup_window");
    if (popWindow) {
        popWindow.parentElement.removeChild(popWindow);
    }
};
