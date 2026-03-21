/**
 * NovelForge – jQuery / Bootstrap client logic
 *
 * Handles:
 *   - Form validation and AJAX submission to /generate_outline
 *   - Rendering and editing of outline / characters
 *   - AJAX POST to /approve_outline
 *   - Starting chapter generation via /generate_chapters
 *   - Polling /progress/<token> to update progress bar
 *   - Export via /export
 */

$(function () {
  "use strict";

  // -------------------------------------------------------------------
  // Bootstrap tooltip initialisation
  // -------------------------------------------------------------------
  $('[data-bs-toggle="tooltip"]').each(function () {
    new bootstrap.Tooltip(this);
  });

  // -------------------------------------------------------------------
  // Step-panel helpers
  // -------------------------------------------------------------------
  var STEPS = ["#step-input", "#step-outline", "#step-progress", "#step-done"];
  var STEP_TAB_BUTTONS = {
    "#step-input": "#step1-novel-setup-btn",
    "#step-outline": "#step2-chapter-outline-btn",
    "#step-progress": "#step3-chapter-writing-btn",
    "#step-done": "#step4-complete-export-btn",
  };

  function showStep(id) {
    var tabButtonSelector = STEP_TAB_BUTTONS[id];
    var tabButton = tabButtonSelector ? document.querySelector(tabButtonSelector) : null;
    if (tabButton) {
      bootstrap.Tab.getOrCreateInstance(tabButton).show();
    }

    // Legacy cleanup: ensure old section-based d-none classes do not hide content.
    $.each(STEPS, function (_, sel) {
      $(sel).removeClass("d-none");
    });
    $("html, body").animate({ scrollTop: 0 }, 200);
  }

  // -------------------------------------------------------------------
  // Alert helpers
  // -------------------------------------------------------------------
  function showAlert(message, type) {
    type = type || "danger";
    var html =
      '<div class="alert alert-' +
      type +
      ' alert-dismissible fade show" role="alert">' +
      '<i class="bi bi-exclamation-triangle-fill me-2"></i>' +
      escapeHtml(message) +
      '<button type="button" class="btn-close" data-bs-dismiss="alert"></button>' +
      "</div>";
    $("#global-alert-area").html(html);
    $("html, body").animate({ scrollTop: 0 }, 200);
  }

  function clearAlerts() {
    $("#global-alert-area").empty();
  }

  // XSS-safe string escaping
  function escapeHtml(str) {
    return $("<div>").text(String(str)).html();
  }

  var DEFAULT_STICKY_STATUS = "AI-Powered Novel Generator";
  var _activeLLMRequests = 0;
  var _hasInitializedLogSnapshot = false;

  function setStickyStatus(text, options) {
    options = options || {};
    if (_activeLLMRequests < 1 && !options.force) {
      return;
    }
    $("#sticky-status-text").text(text || DEFAULT_STICKY_STATUS);
  }

  function inferStatusFromRequestEntry(entry) {
    var messages = (entry && entry.payload && entry.payload.messages) || [];
    var combined = messages
      .map(function (msg) {
        return (msg && msg.content) ? String(msg.content) : "";
      })
      .join("\n")
      .toLowerCase();

      console.log(combined)
    if (!combined) return "Prompting LLM";

    if (combined.indexOf("title") !== -1 && combined.indexOf("novel") !== -1) {
      return "Generating Novel Title";
    }
    if (combined.indexOf("chapter outline") !== -1 || combined.indexOf("chapter-by-chapter") !== -1) {
      return "Generating Chapter Outline";
    }
    if (combined.indexOf("character agent") !== -1 || combined.indexOf("character arc") !== -1) {
      return "Generating Character Arcs";
    }
    if (combined.indexOf("draft agent") !== -1 || combined.indexOf("write the chapter") !== -1) {
      return "Drafting Chapter Content";
    }
    if (combined.indexOf("dialog agent") !== -1 || combined.indexOf("dialogue") !== -1) {
      return "Refining Chapter Dialog";
    }
    if (combined.indexOf("scene agent") !== -1 || combined.indexOf("scene") !== -1) {
      return "Improving Chapter Scenes";
    }
    if (combined.indexOf("context analyzer") !== -1 || combined.indexOf("world-building") !== -1) {
      return "Checking Story Continuity";
    }
    if (combined.indexOf("editing agent") !== -1) {
      return "Editing Chapter Draft";
    }
    if (combined.indexOf("structure agent") !== -1 || combined.indexOf("story architecture") !== -1) {
      return "Validating Story Structure";
    }
    if (combined.indexOf("synthesizer") !== -1) {
      return "Synthesizing Chapter Revisions";
    }
    if (combined.indexOf("polish agent") !== -1 || combined.indexOf("polish") !== -1) {
      return "Polishing Chapter Prose";
    }
    if (combined.indexOf("anti-llm") !== -1 || combined.indexOf("forbidden words") !== -1) {
      return "Removing Robotic Language";
    }
    if (combined.indexOf("quality controller") !== -1 || combined.indexOf("quality control") !== -1) {
      return "Running Quality Control";
    }
    if (combined.indexOf("summary") !== -1 && combined.indexOf("chapter") !== -1) {
      return "Summarizing Chapter";
    }
    if (combined.indexOf("consistency") !== -1) {
      return "Checking Novel Consistency";
    }
    if (combined.indexOf("revise") !== -1 || combined.indexOf("revision") !== -1) {
      return "Applying Chapter Revisions";
    }

    return "Prompting LLM";
  }

  // -------------------------------------------------------------------
  // Progress tracking variables (shared across resume and generation)
  // -------------------------------------------------------------------
  var _pollInterval = null;
  var _progressToken = null;
  var _totalChapters = 0;
  var _doneData = null;

  // -------------------------------------------------------------------
  // Session Resume & New Session
  // -------------------------------------------------------------------
  
  // Check for saved state on page load
  function checkSavedState() {
    $.get("/check_saved_state", function (data) {
      if (data.has_saved_state) {
        // Show resume modal
        $("#resume-title").text(data.title || "Untitled");
        
        var statusText = "";
        if (data.has_progress && data.progress_info) {
          var prog = data.progress_info;
          if (prog.status === "done") {
            statusText = "Generation complete";
          } else if (prog.status === "error") {
            statusText = "Generation stopped with error";
          } else {
            statusText = "In progress - Chapter " + prog.current + " of " + prog.total + " (" + prog.step + ")";
          }
        } else {
          statusText = "Outline ready, generation not started";
        }
        
        $("#resume-status").text(statusText);
        
        // Show the modal
        var modal = new bootstrap.Modal(document.getElementById("resumeModal"));
        modal.show();
      }
    }).fail(function () {
      console.log("No saved state check available");
    });
  }
  
  // Resume button click
  $("#btn-resume").on("click", function () {
    var $btn = $(this);
    $btn.prop("disabled", true).html('<span class="spinner-border spinner-border-sm me-1"></span>Resuming...');
    
    $.post("/resume_session", function (data) {
      // Close modal
      bootstrap.Modal.getInstance(document.getElementById("resumeModal")).hide();
      
      if (data.status === "resumed") {
        // Generation is resuming - show progress step and start polling
        showStep("#step-progress");
        $("#chapter-progress-list").empty();
        _progressToken = data.token;
        _totalChapters = parseInt($("#chapters").val(), 10) || 20;
        _pollInterval = setInterval(pollProgress, 3000);
        // Trigger an immediate poll
        pollProgress();
        showAlert("Resuming chapter generation from where it left off...", "info");
      } else {
        // Just restored session data - reload to show outline
        location.reload();
      }
    }).fail(function () {
      $btn.prop("disabled", false).html('<i class="bi bi-arrow-clockwise me-1"></i>Resume Session');
      showAlert("Failed to resume session. Please try again.", "danger");
    });
  });
  
  // Start fresh button click
  $("#btn-start-fresh").on("click", function () {
    // Just close the modal and stay on the input page
    bootstrap.Modal.getInstance(document.getElementById("resumeModal")).hide();
  });
  
  // New Session button click
  $("#btn-new-session").on("click", function () {
    if (!confirm("Start a new session? This will archive the current progress and clear all data.")) {
      return;
    }
    
    var $btn = $(this);
    $btn.prop("disabled", true);
    
    $.post("/new_session", function () {
      // Reload the page to start fresh
      location.reload();
    }).fail(function () {
      $btn.prop("disabled", false);
      showAlert("Failed to start new session. Please try again.", "danger");
    });
  });
  
  // Check for saved state on page load
  // checkSavedState();

  // -------------------------------------------------------------------
  // Premise character counter
  // -------------------------------------------------------------------
  $("#premise").on("input", function () {
    var len = $(this).val().length;
    $("#premise-count").text(len);
    if (len > 2000) {
      $(this).addClass("is-invalid");
    } else {
      $(this).removeClass("is-invalid");
    }
  });

  // -------------------------------------------------------------------
  // Step 1 – Generate Outline
  // -------------------------------------------------------------------
  $("#novel-form").on("submit", function (e) {
    e.preventDefault();
    clearAlerts();

    // Client-side validation
    var valid = true;

    var premise = $("#premise").val().trim();
    if (!premise || premise.length > 2000) {
      $("#premise").addClass("is-invalid");
      valid = false;
    } else {
      $("#premise").removeClass("is-invalid").addClass("is-valid");
    }

    var genre = $("#genre").val();
    if (!genre) {
      $("#genre").addClass("is-invalid");
      valid = false;
    } else {
      $("#genre").removeClass("is-invalid").addClass("is-valid");
    }

    var chapters = parseInt($("#chapters").val(), 10);
    if (isNaN(chapters) || chapters < 3) {
      $("#chapters").addClass("is-invalid");
      valid = false;
    } else {
      $("#chapters").removeClass("is-invalid").addClass("is-valid");
    }

    var wordCount = parseInt($("#word_count").val(), 10);
    if (isNaN(wordCount) || wordCount < 1000) {
      $("#word_count").addClass("is-invalid");
      valid = false;
    } else {
      $("#word_count").removeClass("is-invalid").addClass("is-valid");
    }

    if (!valid) {
      showAlert("Please fix the validation errors before continuing.", "warning");
      return;
    }

    // Show spinner
    $("#outline-spinner").removeClass("d-none");
    $("#btn-generate-outline").prop("disabled", true);

    var payload = {
      premise: premise,
      genre: genre,
      chapters: chapters,
      word_count: wordCount,
      special_events: $("#special_events").val().trim(),
      special_instructions: $("#special_instructions").val().trim(),
    };

    $.ajax({
      url: "/generate_outline",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify(payload),
      success: function (resp) {
        renderOutline(resp);
        showStep("#step-outline");
      },
      error: function (xhr) {
        var msg =
          (xhr.responseJSON && xhr.responseJSON.error) ||
          "Failed to generate outline. Check your LLM API configuration.";
        showAlert(msg);
      },
      complete: function () {
        $("#outline-spinner").addClass("d-none");
        $("#btn-generate-outline").prop("disabled", false);
      },
    });
  });

  // -------------------------------------------------------------------
  // Render outline into the review table
  // -------------------------------------------------------------------
  function renderOutline(data) {
    $("#outline-title").val(data.title || "");
    

    // Chapters table
    var $tbody = $("#chapter-tbody").empty();
    $.each(data.chapters || [], function (_, ch) {
      addChapterRow(ch.number || "", ch.title || "", ch.summary || "");
    });

    // Characters table
    var $ctbody = $("#characters-tbody").empty();
    $.each(data.characters || [], function (_, c) {
      addCharacterRow(c.name || "", c.age || "", c.role || "", c.background || "", c.arc || "");
    });
  }

  // -------------------------------------------------------------------
  // Add/Delete Character Functions
  // -------------------------------------------------------------------
  function addCharacterRow(name, age, role, background, arc) {
    var row =
      "<tr>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='name'>" + escapeHtml(name) + "</div></td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='age'>" + escapeHtml(age) + "</div></td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='role'>" + escapeHtml(role) + "</div></td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='background'>" + escapeHtml(background) + "</div></td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='arc'>" + escapeHtml(arc) + "</div></td>" +
      "<td class='text-center'>" +
      "<button class='btn btn-sm btn-outline-danger btn-delete-character' title='Delete Character'><i class='bi bi-trash'></i></button>" +
      "</td>" +
      "</tr>";
    $("#characters-tbody").append(row);
  }

  // Add Character button
  $("#btn-add-character").on("click", function () {
    addCharacterRow("New Character", "", "Protagonist/Antagonist/Supporting", "Enter background...", "Enter character arc...");
  });

  // Delete Character button (delegated event)
  $("#characters-tbody").on("click", ".btn-delete-character", function () {
    var $row = $(this).closest("tr");
    var characterName = $row.find("[data-field='name']").text().trim();
    
    if ($("#characters-tbody tr").length <= 1) {
      showAlert("Cannot delete the last character. At least one character is required.", "warning");
      return;
    }
    
    if (confirm("Delete character '" + characterName + "'?")) {
      $row.remove();
    }
  });

  // -------------------------------------------------------------------
  // Add/Delete Chapter Functions
  // -------------------------------------------------------------------
  function addChapterRow(number, title, summary) {
    var row =
      "<tr>" +
      "<td class='chapter-number'>" + escapeHtml(number) + "</td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='title'>" +
      escapeHtml(title) +
      "</div></td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='summary'>" +
      escapeHtml(summary) +
      "</div></td>" +
      "<td class='text-center'>" +
      "<div class='btn-group btn-group-sm me-1' role='group'>" +
      "<button class='btn btn-outline-secondary btn-move-up' title='Move Up'><i class='bi bi-arrow-up'></i></button>" +
      "<button class='btn btn-outline-secondary btn-move-down' title='Move Down'><i class='bi bi-arrow-down'></i></button>" +
      "</div>" +
      "<div class='btn-group btn-group-sm me-1' role='group'>" +
      "<button class='btn btn-outline-success btn-add-before' title='Add Before'><i class='bi bi-plus-circle'></i></button>" +
      "<button class='btn btn-outline-success btn-add-after' title='Add After'><i class='bi bi-plus-circle'></i></button>" +
      "</div>" +
      "<button class='btn btn-sm btn-outline-danger btn-delete-chapter' title='Delete Chapter'><i class='bi bi-trash'></i></button>" +
      "</td>" +
      "</tr>";
    $("#chapter-tbody").append(row);
    renumberChapters();
  }

  function renumberChapters() {
    $("#chapter-tbody tr").each(function (idx) {
      $(this).find(".chapter-number").text(idx + 1);
    });
  }

  // Move chapter up
  $("#chapter-tbody").on("click", ".btn-move-up", function () {
    var $row = $(this).closest("tr");
    var $prev = $row.prev();
    if ($prev.length) {
      $row.insertBefore($prev);
      renumberChapters();
    }
  });

  // Move chapter down
  $("#chapter-tbody").on("click", ".btn-move-down", function () {
    var $row = $(this).closest("tr");
    var $next = $row.next();
    if ($next.length) {
      $row.insertAfter($next);
      renumberChapters();
    }
  });

  // Add chapter before
  $("#chapter-tbody").on("click", ".btn-add-before", function () {
    var $row = $(this).closest("tr");
    var newRow =
      "<tr>" +
      "<td class='chapter-number'></td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='title'>New Chapter</div></td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='summary'>Enter chapter summary...</div></td>" +
      "<td class='text-center'>" +
      "<div class='btn-group btn-group-sm me-1' role='group'>" +
      "<button class='btn btn-outline-secondary btn-move-up' title='Move Up'><i class='bi bi-arrow-up'></i></button>" +
      "<button class='btn btn-outline-secondary btn-move-down' title='Move Down'><i class='bi bi-arrow-down'></i></button>" +
      "</div>" +
      "<div class='btn-group btn-group-sm me-1' role='group'>" +
      "<button class='btn btn-outline-success btn-add-before' title='Add Before'><i class='bi bi-plus-circle'></i></button>" +
      "<button class='btn btn-outline-success btn-add-after' title='Add After'><i class='bi bi-plus-circle'></i></button>" +
      "</div>" +
      "<button class='btn btn-sm btn-outline-danger btn-delete-chapter' title='Delete Chapter'><i class='bi bi-trash'></i></button>" +
      "</td>" +
      "</tr>";
    $row.before(newRow);
    renumberChapters();
  });

  // Add chapter after
  $("#chapter-tbody").on("click", ".btn-add-after", function () {
    var $row = $(this).closest("tr");
    var newRow =
      "<tr>" +
      "<td class='chapter-number'></td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='title'>New Chapter</div></td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='summary'>Enter chapter summary...</div></td>" +
      "<td class='text-center'>" +
      "<div class='btn-group btn-group-sm me-1' role='group'>" +
      "<button class='btn btn-outline-secondary btn-move-up' title='Move Up'><i class='bi bi-arrow-up'></i></button>" +
      "<button class='btn btn-outline-secondary btn-move-down' title='Move Down'><i class='bi bi-arrow-down'></i></button>" +
      "</div>" +
      "<div class='btn-group btn-group-sm me-1' role='group'>" +
      "<button class='btn btn-outline-success btn-add-before' title='Add Before'><i class='bi bi-plus-circle'></i></button>" +
      "<button class='btn btn-outline-success btn-add-after' title='Add After'><i class='bi bi-plus-circle'></i></button>" +
      "</div>" +
      "<button class='btn btn-sm btn-outline-danger btn-delete-chapter' title='Delete Chapter'><i class='bi bi-trash'></i></button>" +
      "</td>" +
      "</tr>";
    $row.after(newRow);
    renumberChapters();
  });

  // Delete Chapter button (delegated event)
  $("#chapter-tbody").on("click", ".btn-delete-chapter", function () {
    var $row = $(this).closest("tr");
    var chapterNum = $row.find(".chapter-number").text();
    
    if ($("#chapter-tbody tr").length <= 1) {
      showAlert("Cannot delete the last chapter. At least one chapter is required.", "warning");
      return;
    }
    
    if (confirm("Delete Chapter " + chapterNum + "?")) {
      $row.remove();
      renumberChapters();
    }
  });

  // -------------------------------------------------------------------
  // Back button on outline step
  // -------------------------------------------------------------------
  $("#btn-back-to-input").on("click", function () {
    showStep("#step-input");
  });

  // -------------------------------------------------------------------
  // Step 2 – Approve Outline
  // -------------------------------------------------------------------
  $("#btn-approve-outline").on("click", function () {
    clearAlerts();

    var title = $("#outline-title").val().trim();
    if (!title) {
      showAlert("Title cannot be empty.", "warning");
      return;
    }

    // Collect edited chapters from table
    var chapters = [];
    $("#chapter-tbody tr").each(function (idx) {
      var $row = $(this);
      chapters.push({
        number: idx + 1,
        title: $row.find("[data-field='title']").text().trim(),
        summary: $row.find("[data-field='summary']").text().trim(),
      });
    });

    // Collect edited characters
    var characters = [];
    $("#characters-tbody tr").each(function () {
      var $row = $(this);
      characters.push({
        name: $row.find("[data-field='name']").text().trim(),
        age: $row.find("[data-field='age']").text().trim(),
        role: $row.find("[data-field='role']").text().trim(),
        background: $row.find("[data-field='background']").text().trim(),
        arc: $row.find("[data-field='arc']").text().trim(),
      });
    });

    $("#approve-spinner").removeClass("d-none");
    $("#btn-approve-outline").prop("disabled", true);

    // Show the step 3 - chapter writing tab
    showStep("#step3-chapter-writing-tab");

    $.ajax({
      url: "/approve_outline",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ title: title, chapters: chapters, characters: characters }),
      success: function () {
        startChapterGeneration();
      },
      error: function (xhr) {
        var msg = (xhr.responseJSON && xhr.responseJSON.error) || "Failed to save outline.";
        showAlert(msg);
      },
      complete: function () {
        $("#approve-spinner").addClass("d-none");
        $("#btn-approve-outline").prop("disabled", false);
      },
    });
  });

  // -------------------------------------------------------------------
  // Step 3 – Chapter Generation
  // -------------------------------------------------------------------

  function startChapterGeneration() {
    showStep("#step-progress");
    $("#chapter-progress-list").empty();
    updateProgressBar(0, 0, "Preparing…");

    $.ajax({
      url: "/generate_chapters",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({}),
      success: function (resp) {
        _progressToken = resp.token;
        _totalChapters = parseInt($("#chapters").val(), 10) || 20;
        _pollInterval = setInterval(pollProgress, 3000);
      },
      error: function (xhr) {
        var msg = (xhr.responseJSON && xhr.responseJSON.error) || "Failed to start generation.";
        showAlert(msg);
        showStep("#step-outline");
      },
    });
  }

  function pollProgress() {
    if (!_progressToken) return;

    $.ajax({
      url: "/progress/" + _progressToken,
      method: "GET",
      success: function (data) {
        var current = data.current || 0;
        var total = data.total || _totalChapters;

        updateProgressBar(current, total, data.step || null);

        // Render completed chapters in list
        var $list = $("#chapter-progress-list");
        $list.empty();
        $.each(data.chapters_done || [], function (_, ch) {
          $list.append(
            '<li class="list-group-item done-chapter">' +
            '<i class="bi bi-check-circle-fill text-success me-2"></i>' +
            "Chapter " + escapeHtml(ch.number) + ": " + escapeHtml(ch.title) +
            "</li>"
          );
        });

        if (data.status === "done") {
          clearInterval(_pollInterval);
          showDoneStep(data);
        } else if (data.status === "error") {
          clearInterval(_pollInterval);
          showAlert("Chapter generation failed: " + (data.error || "Unknown error"));
          showStep("#step-outline");
        }
      },
      error: function () {
        // Non-fatal – keep polling
      },
    });
  }

  function updateProgressBar(current, total, step) {
    var pct = total > 0 ? Math.round((current / total) * 100) : 0;
    $("#progress-bar").css("width", pct + "%").attr("aria-valuenow", pct);
    $("#progress-percent").text(pct + "%");
    if (step !== null && step !== undefined && step !== "") {
      $("#progress-label").text(step);
    } else if (current < total) {
      $("#progress-label").text("Writing chapter " + (current + 1) + " of " + total + "…");
    } else {
      $("#progress-label").text("Finalising…");
    }
  }

  // -------------------------------------------------------------------
  // Step 4 – Done
  // -------------------------------------------------------------------
  function showDoneStep(data) {
    _doneData = data || {};

    var title = $("#outline-title").val() || "Your Novel";
    var chaptersCount = (data.chapters_done || []).length;
    var wordEst = chaptersCount * Math.round(parseInt($("#word_count").val(), 10) / (parseInt($("#chapters").val(), 10) || 20));

    $("#done-title").text(title);
    $("#done-stats").text(
      chaptersCount + " chapters written · ~" + wordEst.toLocaleString() + " words"
    );

    // Consistency notes
    var consistency = data.consistency || {};
    if (consistency.overall_assessment || (consistency.issues && consistency.issues.length)) {
      $("#consistency-alert").removeClass("d-none");
      $("#consistency-assessment").text(consistency.overall_assessment || "");
      var $ul = $("#consistency-issues").empty();
      $.each(consistency.issues || [], function (_, issue) {
        $ul.append("<li>" + escapeHtml(issue) + "</li>");
      });
    } else {
      $("#consistency-alert").addClass("d-none");
      $("#consistency-assessment").text("");
      $("#consistency-issues").empty();
    }

    // Populate chapter revision selector
    var $reviseSelect = $("#revise-chapter-select").empty();
    $.each(data.chapters_done || [], function (_, ch) {
      $reviseSelect.append(
        $("<option>")
          .val(ch.number)
          .text("Chapter " + ch.number + ": " + (ch.title || "Untitled"))
      );
    });
    var hasChapters = (data.chapters_done || []).length > 0;
    $("#btn-revise-chapter").prop("disabled", !hasChapters);

    // Build preview accordion
    var $acc = $("#chapters-preview-accordion").empty();
    $.each(data.chapters_done || [], function (i, ch) {
      var id = "ch-accordion-" + i;
      var heading = "ch-heading-" + i;
      $acc.append(
        '<div class="accordion-item">' +
        '<h2 class="accordion-header" id="' + heading + '">' +
        '<button class="accordion-button collapsed" type="button" ' +
        'data-bs-toggle="collapse" data-bs-target="#' + id + '" ' +
        'aria-expanded="false" aria-controls="' + id + '">' +
        "Chapter " + escapeHtml(ch.number) + ": " + escapeHtml(ch.title) +
        "</button></h2>" +
        '<div id="' + id + '" class="accordion-collapse collapse" aria-labelledby="' + heading + '">' +
        '<div class="accordion-body"><pre class="mb-0 text-wrap">' + escapeHtml(ch.content || "") + "</pre></div>" +
        "</div></div>"
      );
    });

    showStep("#step-done");
  }

  // -------------------------------------------------------------------
  // Export
  // -------------------------------------------------------------------
  $("#btn-export").on("click", function () {
    clearAlerts();
    $("#export-spinner").removeClass("d-none");
    $("#btn-export").prop("disabled", true);

    $.ajax({
      url: "/export",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ token: _progressToken }),
      success: function (resp) {
        if (resp.download_url) {
          // Create a temporary link and trigger download
          var $a = $("<a>")
            .attr("href", resp.download_url)
            .attr("download", "")
            .appendTo("body");
          $a[0].click();
          $a.remove();
        }
      },
      error: function (xhr) {
        var msg = (xhr.responseJSON && xhr.responseJSON.error) || "Export failed.";
        showAlert(msg);
      },
      complete: function () {
        $("#export-spinner").addClass("d-none");
        $("#btn-export").prop("disabled", false);
      },
    });
  });

  $("#btn-export-editors-notes").on("click", function () {
    clearAlerts();
    $("#export-editors-notes-spinner").removeClass("d-none");
    $("#btn-export-editors-notes").prop("disabled", true);

    $.ajax({
      url: "/export_editors_notes",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ token: _progressToken }),
      success: function (resp) {
        if (resp.download_url) {
          var $a = $("<a>")
            .attr("href", resp.download_url)
            .attr("download", "")
            .appendTo("body");
          $a[0].click();
          $a.remove();
        }
      },
      error: function (xhr) {
        var msg =
          (xhr.responseJSON && xhr.responseJSON.error) ||
          "Editor's notes export failed.";
        showAlert(msg);
      },
      complete: function () {
        $("#export-editors-notes-spinner").addClass("d-none");
        $("#btn-export-editors-notes").prop("disabled", false);
      },
    });
  });

  $("#btn-revise-chapter").on("click", function () {
    clearAlerts();

    if (!_progressToken) {
      showAlert("No active generation token was found. Please regenerate chapters.", "warning");
      return;
    }

    var chapterNumber = parseInt($("#revise-chapter-select").val(), 10);
    var instructions = $("#revise-instructions").val().trim();

    if (isNaN(chapterNumber) || chapterNumber < 1) {
      showAlert("Please select a chapter to revise.", "warning");
      return;
    }

    if (!instructions) {
      showAlert("Please enter revision instructions before applying.", "warning");
      return;
    }

    $("#revise-chapter-spinner").removeClass("d-none");
    $("#btn-revise-chapter").prop("disabled", true);

    $.ajax({
      url: "/revise_chapter",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({
        token: _progressToken,
        chapter_number: chapterNumber,
        instructions: instructions,
      }),
      success: function (resp) {
        _doneData = resp;
        showDoneStep(_doneData);
        $("#revise-instructions").val("");
        showAlert("Chapter revision complete. All chapter agents were rerun.", "success");
      },
      error: function (xhr) {
        var msg =
          (xhr.responseJSON && xhr.responseJSON.error) ||
          "Chapter revision failed.";
        showAlert(msg);
      },
      complete: function () {
        $("#revise-chapter-spinner").addClass("d-none");
        $("#btn-revise-chapter").prop("disabled", false);
      },
    });
  });

  // -------------------------------------------------------------------
  // LLM Log Display
  // -------------------------------------------------------------------
  var _seenLogSignatures = {};
  var _logPollInterval = null;

  function entrySignature(entry) {
    if (!entry) return "";
    var payloadMessages = (entry.payload && entry.payload.messages) || [];
    var messageCount = payloadMessages.length;
    var firstMessage = messageCount > 0 && payloadMessages[0].content ? String(payloadMessages[0].content).slice(0, 60) : "";
    return [entry.timestamp || "", entry.type || "", messageCount, firstMessage].join("|");
  }

  function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + "...";
  }

  function formatLogEntry(entry) {
    if (entry.type === "request") {
      // Format request with messages
      var content = "";
      if (entry.payload && entry.payload.messages) {
        entry.payload.messages.forEach(function(msg) {
          if (msg.role === "system") {
            content += "[System] " + truncateText(msg.content, 500) + "\n\n";
          } else if (msg.role === "user") {
            content += truncateText(msg.content, 2500);
          }
        });
      }
      return {
        type: "request",
        header: "Request to LLM",
        content: content,
        timestamp: entry.timestamp
      };
    } else if (entry.type === "response") {
      // Format response
      var content = "";
      if (entry.response && entry.response.choices && entry.response.choices[0]) {
        var message = entry.response.choices[0].message;
        if (message && message.content) {
          content = truncateText(message.content, 2500);
        }
      }
      return {
        type: "response",
        header: "LLM Response",
        content: content,
        timestamp: entry.timestamp
      };
    }
    return null;
  }

  function addLogMessage(formatted) {
    if (!formatted || !formatted.content) return;

    var messageClass = formatted.type === "request" ? "request" : "response";
    var html = 
      '<div class="llm-message ' + messageClass + '">' +
        '<div class="llm-bubble">' +
          '<div class="llm-bubble-header">' + escapeHtml(formatted.header) + '</div>' +
          '<div class="llm-bubble-content">' + escapeHtml(formatted.content) + '</div>' +
          '<div class="llm-bubble-timestamp">' + escapeHtml(formatted.timestamp) + '</div>' +
        '</div>' +
      '</div>';
    
    $("#llm-chat-messages").append(html);
    
    // Auto-scroll to bottom
    var chatWindow = document.getElementById("llm-chat-window");
    if (chatWindow) {
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }
  }

  function pollLLMLog() {
    $.ajax({
      url: "/llm_log",
      method: "GET",
      success: function(data) {
        var entries = data.entries || [];

        if (!_hasInitializedLogSnapshot) {
          for (var j = 0; j < entries.length; j++) {
            var initialSignature = entrySignature(entries[j]);
            if (initialSignature) {
              _seenLogSignatures[initialSignature] = true;
            }
          }
          _hasInitializedLogSnapshot = true;
          _activeLLMRequests = 0;
          setStickyStatus(DEFAULT_STICKY_STATUS, { force: true });
          return;
        }

        if (entries.length === 0) return;

        // Clear placeholder on first visible log entry
        if (Object.keys(_seenLogSignatures).length === 0) {
          $("#llm-chat-messages").empty();
        }

        for (var i = 0; i < entries.length; i++) {
          var entry = entries[i];
          var signature = entrySignature(entry);
          if (!signature || _seenLogSignatures[signature]) {
            continue;
          }

          _seenLogSignatures[signature] = true;

          if (entry.type === "request") {
            _activeLLMRequests += 1;
            setStickyStatus(inferStatusFromRequestEntry(entry));
          } else if (entry.type === "response" || entry.type === "error") {
            if (_activeLLMRequests > 0) {
              _activeLLMRequests -= 1;
            }
            if (_activeLLMRequests === 0) {
              setStickyStatus(DEFAULT_STICKY_STATUS, { force: true });
            }
          }

          var formatted = formatLogEntry(entry);
          if (formatted) {
            addLogMessage(formatted);
          }
        }
      },
      error: function() {
        // Silently fail - log polling is non-critical
      }
    });
  }

  // Start polling for LLM log updates
  _logPollInterval = setInterval(pollLLMLog, 5000); // Poll every 5 seconds
  pollLLMLog(); // Initial poll

  // Clear log button
  $("#btn-clear-log").on("click", function() {
    $("#llm-chat-messages").html(
      '<div class="text-center text-muted small py-3">' +
      '<i class="bi bi-info-circle me-1"></i>Log cleared (still polling)' +
      '</div>'
    );
    _seenLogSignatures = {};
    _activeLLMRequests = 0;
    _hasInitializedLogSnapshot = false;
    setStickyStatus(DEFAULT_STICKY_STATUS, { force: true });
  });

  // -------------------------------------------------------------------
  // Start Over
  // -------------------------------------------------------------------
  $("#btn-start-over").on("click", function () {
    clearAlerts();
    clearInterval(_pollInterval);
    _progressToken = null;

    // Reset form
    $("#novel-form")[0].reset();
    $("#premise-count").text("0");
    $(".is-valid, .is-invalid").removeClass("is-valid is-invalid");

    // Clear generated content
    $("#consistency-alert").addClass("d-none");
    $("#consistency-assessment").text("");
    $("#consistency-issues").empty();
    $("#chapters-preview-accordion").empty();
    $("#chapter-progress-list").empty();
    $("#revise-instructions").val("");
    $("#revise-chapter-select").empty();
    _doneData = null;

    showStep("#step-input");
  });
});
