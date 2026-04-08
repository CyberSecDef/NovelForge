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

// -------------------------------------------------------------------
// Theme toggle – runs before DOM ready to prevent flash of wrong theme
// -------------------------------------------------------------------
(function () {
  var saved = localStorage.getItem("nf-theme");
  var prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  var theme = saved || (prefersDark ? "dark" : "light");
  document.documentElement.setAttribute("data-bs-theme", theme);
})();

$(function () {
  "use strict";

  // -------------------------------------------------------------------
  // Theme toggle button
  // -------------------------------------------------------------------
  function applyTheme(theme) {
    document.documentElement.setAttribute("data-bs-theme", theme);
    localStorage.setItem("nf-theme", theme);
    var $icon = $("#theme-icon");
    if (theme === "dark") {
      $icon.removeClass("bi-moon-fill").addClass("bi-sun-fill");
    } else {
      $icon.removeClass("bi-sun-fill").addClass("bi-moon-fill");
    }
  }

  // Set icon to match current theme on load
  applyTheme(document.documentElement.getAttribute("data-bs-theme") || "light");

  $("#btn-toggle-theme").on("click", function () {
    var current = document.documentElement.getAttribute("data-bs-theme");
    applyTheme(current === "dark" ? "light" : "dark");
  });

  // -------------------------------------------------------------------
  // CSRF token – attach to every AJAX request via X-CSRFToken header
  // Read from cookie first (survives page refresh), fall back to meta tag
  // -------------------------------------------------------------------
  function getCsrfToken() {
    var match = document.cookie.match(/(^|;\s*)csrf_token=([^;]+)/);
    if (match) return decodeURIComponent(match[2]);
    return $('meta[name="csrf-token"]').attr("content") || "";
  }
  $.ajaxSetup({
    beforeSend: function (xhr, settings) {
      if (!/^(GET|HEAD|OPTIONS)$/i.test(settings.type)) {
        var token = getCsrfToken();
        if (token) {
          xhr.setRequestHeader("X-CSRFToken", token);
        }
      }
    },
  });

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

  // -------------------------------------------------------------------
  // Contenteditable field length limits
  // -------------------------------------------------------------------
  var FIELD_MAX_LENGTHS = {
    name: 100,
    age: 50,
    role: 200,
    title: 200,
    summary: 2000,
    background: 2000,
    arc: 2000,
  };

  $(document).on("input", ".editable-cell[contenteditable]", function () {
    var $el = $(this);
    var field = $el.data("field");
    var max = FIELD_MAX_LENGTHS[field];
    if (max && $el.text().length > max) {
      var sel = window.getSelection();
      var offset = sel.rangeCount ? sel.getRangeAt(0).startOffset : 0;
      $el.text($el.text().substring(0, max));
      // Restore cursor position
      try {
        var range = document.createRange();
        var node = $el[0].firstChild;
        if (node) {
          range.setStart(node, Math.min(offset, node.length));
          range.collapse(true);
          sel.removeAllRanges();
          sel.addRange(range);
        }
      } catch (e) { /* ignore cursor restore errors */ }
    }
  });

  // -------------------------------------------------------------------
  // Unsaved changes tracking for Step 2 outline/character edits
  // -------------------------------------------------------------------
  var _outlineDirty = false;

  function markOutlineDirty() {
    _outlineDirty = true;
  }

  function clearOutlineDirty() {
    _outlineDirty = false;
  }

  // Contenteditable cell edits
  $(document).on("input", "#chapter-tbody .editable-cell, #characters-tbody .editable-cell", markOutlineDirty);

  // When a character name is edited, sync the perspective dropdown
  $(document).on("input", "#characters-tbody .editable-cell[data-field='name']", function () {
    syncPerspectiveDropdown();
  });

  // Outline title edits
  $(document).on("input", "#outline-title", markOutlineDirty);

  // Browser/tab close warning
  $(window).on("beforeunload", function () {
    if (_outlineDirty) {
      return "You have unsaved changes to the outline. Leave without approving?";
    }
  });

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
  // Session Management: Dropdown, New Session, Delete Session
  // -------------------------------------------------------------------

  // Populate sessions dropdown on page load and when dropdown is opened
  function loadSessionsList() {
    $.get("/list_sessions", function (data) {
      var $menu = $("#sessions-dropdown-menu");
      $menu.empty();

      var sessions = data.sessions || [];
      if (sessions.length === 0) {
        $menu.append('<li><span class="dropdown-item text-muted">No saved sessions</span></li>');
        return;
      }

      $.each(sessions, function (_, s) {
        var $item = $("<li>");
        var $link = $('<button class="dropdown-item" type="button"></button>');
        $link.text(s.title);
        $link.attr("data-session-id", s.session_id);
        $link.on("click", function () {
          loadSession(s.session_id);
        });
        $item.append($link);
        $menu.append($item);
      });
    }).fail(function () {
      var $menu = $("#sessions-dropdown-menu");
      $menu.empty();
      $menu.append('<li><span class="dropdown-item text-muted">Failed to load sessions</span></li>');
    });
  }

  // Load a specific session by ID
  function loadSession(sessionId) {
    if (_outlineDirty && !confirm("You have unsaved outline changes. Discard and load a different session?")) {
      return;
    }
    clearOutlineDirty(); // prevent beforeunload firing during reload
    $.ajax({
      url: "/load_session",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ session_id: sessionId }),
      success: function () {
        location.reload();
      },
      error: function (xhr) {
        var msg = (xhr.responseJSON && xhr.responseJSON.error) || "Failed to load session.";
        showAlert(msg);
      },
    });
  }

  // Refresh the dropdown each time it is opened
  $("#btn-sessions-dropdown").on("show.bs.dropdown", function () {
    loadSessionsList();
  });

  // New Session button click
  $("#btn-new-session").on("click", function () {
    var msg = _outlineDirty
      ? "You have unsaved outline changes. Start a new session anyway? This will archive current progress and clear all data."
      : "Start a new session? This will archive the current progress and clear all data.";
    if (!confirm(msg)) {
      return;
    }
    clearOutlineDirty();

    var $btn = $(this);
    $btn.prop("disabled", true);

    $.post("/new_session", function () {
      location.reload();
    }).fail(function () {
      $btn.prop("disabled", false);
      showAlert("Failed to start new session. Please try again.", "danger");
    });
  });

  // Delete Session button click
  $("#btn-delete-session").on("click", function () {
    var msg = _outlineDirty
      ? "You have unsaved outline changes. Delete the current session anyway? This cannot be undone."
      : "Delete the current session? This cannot be undone.";
    if (!confirm(msg)) {
      return;
    }
    clearOutlineDirty();

    var $btn = $(this);
    $btn.prop("disabled", true);

    $.post("/delete_session", function () {
      location.reload();
    }).fail(function () {
      $btn.prop("disabled", false);
      showAlert("Failed to delete session. Please try again.", "danger");
    });
  });

  // Load the sessions list on initial page load
  loadSessionsList();

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

    // Sync perspective dropdown with newly rendered characters
    syncPerspectiveDropdown();

    // Freshly rendered outline is clean — no unsaved changes
    clearOutlineDirty();
  }

  // -------------------------------------------------------------------
  // Add/Delete Character Functions
  // -------------------------------------------------------------------
  function addCharacterRow(name, age, role, background, arc) {
    var safeName = escapeHtml(name);
    var row =
      "<tr>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='name' role='textbox' aria-label='Character name'>" + safeName + "</div></td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='age' role='textbox' aria-label='Age for " + safeName + "'>" + escapeHtml(age) + "</div></td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='role' role='textbox' aria-label='Role for " + safeName + "'>" + escapeHtml(role) + "</div></td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='background' role='textbox' aria-label='Background for " + safeName + "'>" + escapeHtml(background) + "</div></td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='arc' role='textbox' aria-label='Arc for " + safeName + "'>" + escapeHtml(arc) + "</div></td>" +
      "<td class='text-center'>" +
      "<button class='btn btn-sm btn-outline-danger btn-delete-character' title='Delete Character' aria-label='Delete " + safeName + "'><i class='bi bi-trash'></i></button>" +
      "</td>" +
      "</tr>";
    $("#characters-tbody").append(row);
  }

  // Sync the Narrative Perspective dropdown with current character names
  function syncPerspectiveDropdown() {
    var $select = $("#narrative-perspective");
    var currentVal = $select.val();
    // Remove all first-person options (keep third_person)
    $select.find("option[value!='third_person']").remove();
    // Add an option for each character
    $("#characters-tbody tr").each(function () {
      var name = $(this).find("[data-field='name']").text().trim();
      if (name) {
        $select.append(
          $("<option></option>").val("first_person:" + name).text("First Person – " + name)
        );
      }
    });
    // Restore previous selection if it still exists
    if ($select.find("option[value='" + currentVal + "']").length) {
      $select.val(currentVal);
    } else {
      $select.val("third_person");
    }
  }

  // Update perspective dropdown whenever characters change
  $("#narrative-perspective").on("change", function () {
    markOutlineDirty();
  });

  // Add Character button
  $("#btn-add-character").on("click", function () {
    addCharacterRow("New Character", "", "Protagonist/Antagonist/Supporting", "Enter background...", "Enter character arc...");
    markOutlineDirty();
    syncPerspectiveDropdown();
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
      markOutlineDirty();
      syncPerspectiveDropdown();
    }
  });

  // -------------------------------------------------------------------
  // Add/Delete Chapter Functions
  // -------------------------------------------------------------------
  function addChapterRow(number, title, summary) {
    var chLabel = "Chapter " + escapeHtml(number);
    var row =
      "<tr>" +
      "<td class='chapter-number'>" + escapeHtml(number) + "</td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='title' role='textbox' aria-label='" + chLabel + " title'>" +
      escapeHtml(title) +
      "</div></td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='summary' role='textbox' aria-label='" + chLabel + " summary'>" +
      escapeHtml(summary) +
      "</div></td>" +
      "<td class='text-center'>" +
      "<div class='btn-group btn-group-sm me-1' role='group' aria-label='Reorder " + chLabel + "'>" +
      "<button class='btn btn-outline-secondary btn-move-up' title='Move Up' aria-label='Move " + chLabel + " up'><i class='bi bi-arrow-up'></i></button>" +
      "<button class='btn btn-outline-secondary btn-move-down' title='Move Down' aria-label='Move " + chLabel + " down'><i class='bi bi-arrow-down'></i></button>" +
      "</div>" +
      "<div class='btn-group btn-group-sm me-1' role='group' aria-label='Insert around " + chLabel + "'>" +
      "<button class='btn btn-outline-success btn-add-before' title='Add Before' aria-label='Add chapter before " + chLabel + "'><i class='bi bi-plus-circle'></i></button>" +
      "<button class='btn btn-outline-success btn-add-after' title='Add After' aria-label='Add chapter after " + chLabel + "'><i class='bi bi-plus-circle'></i></button>" +
      "</div>" +
      "<button class='btn btn-sm btn-outline-danger btn-delete-chapter' title='Delete Chapter' aria-label='Delete " + chLabel + "'><i class='bi bi-trash'></i></button>" +
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
      markOutlineDirty();
    }
  });

  // Move chapter down
  $("#chapter-tbody").on("click", ".btn-move-down", function () {
    var $row = $(this).closest("tr");
    var $next = $row.next();
    if ($next.length) {
      $row.insertAfter($next);
      renumberChapters();
      markOutlineDirty();
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
    markOutlineDirty();
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
    markOutlineDirty();
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
      markOutlineDirty();
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

    var narrativePerspective = $("#narrative-perspective").val() || "third_person";

    $("#approve-spinner").removeClass("d-none");
    $("#btn-approve-outline").prop("disabled", true);

    // Show the step 3 - chapter writing tab
    showStep("#step3-chapter-writing-tab");

    $.ajax({
      url: "/approve_outline",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ title: title, chapters: chapters, characters: characters, narrative_perspective: narrativePerspective }),
      success: function () {
        clearOutlineDirty();
        startChapterGeneration();
      },
      error: function (xhr) {
        var msg = (xhr.responseJSON && xhr.responseJSON.error) || "Failed to save outline. The AI service may be unavailable — your edits are preserved.";
        showAlert(msg);
        // Re-enable the button so the user can try again
        $("#btn-approve-outline").prop("disabled", false);
      },
      complete: function () {
        $("#approve-spinner").addClass("d-none");
        // Button stays disabled while chapter generation is running;
        // it will be re-enabled when generation completes or errors.
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
    // Ensure the approve button stays disabled while generation runs
    $("#btn-approve-outline").prop("disabled", true);

    $.ajax({
      url: "/generate_chapters",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({}),
      success: function (resp) {
        _progressToken = resp.token;
        _totalChapters = parseInt($("#chapters").val(), 10) || 20;
        _pollDelay = _pollDelayMin;
        _lastPollStep = "";
        _pollFailures = 0;
        _lastCompletedCount = 0;
        _chapterCompletionTimes = [];
        _generationStartTime = Date.now();
        _latestFullData = {};
        _lastFullFetchTime = 0;
        $("#progress-time-estimate").addClass("d-none").text("");
        _schedulePoll();
      },
      error: function (xhr) {
        var json = xhr.responseJSON || {};
        if (xhr.status === 409 && json.error_code === "generation_in_progress") {
          // A generation is already running — attach to it and start polling
          _progressToken = json.token;
          _totalChapters = parseInt($("#chapters").val(), 10) || 20;
          _pollDelay = _pollDelayMin;
          _lastPollStep = "";
          _pollFailures = 0;
          _lastCompletedCount = 0;
          _chapterCompletionTimes = [];
          _generationStartTime = Date.now();
          _latestFullData = {};
          _lastFullFetchTime = 0;
          $("#progress-time-estimate").addClass("d-none").text("");
          _schedulePoll();
        } else {
          var msg = json.error || "Failed to start chapter generation. Please check your connection and try again.";
          showAlert(msg);
          showStep("#step-outline");
          // Re-enable button so the user can retry
          $("#btn-approve-outline").prop("disabled", false);
        }
      },
    });
  }

  var _pollFailures = 0;
  var _pollDelay = 15000;       // current adaptive delay (ms)
  var _pollDelayMin = 15000;    // 15s floor
  var _pollDelayMax = 60000;    // 60s ceiling
  var _lastPollStep = "";       // last observed step label

  // Full-data fetch state
  var _latestFullData = {};          // most recent full payload from /progress/<token>/full
  var _lastFullFetchTime = 0;        // ms timestamp of the last full fetch
  var _fullFetchIntervalMs = 120000; // fetch full data every 2 minutes

  // Time estimation state
  var _lastCompletedCount = 0;           // chapters completed at last poll
  var _chapterCompletionTimes = [];      // timestamps (ms) when each chapter finished
  var _generationStartTime = 0;          // when generation began

  function _schedulePoll() {
    _pollInterval = setTimeout(pollProgress, _pollDelay);
  }

  /**
   * Fetch the full heavyweight payload from /progress/<token>/full.
   * Updates _latestFullData, refreshes the chapter list, and calls
   * showDoneStep if generation has finished.
   */
  function fetchFullProgress(onComplete) {
    if (!_progressToken) return;
    $.ajax({
      url: "/progress/" + _progressToken + "/full",
      method: "GET",
      success: function (fullData) {
        _latestFullData = fullData || {};
        _lastFullFetchTime = Date.now();

        // Refresh the chapter list with authoritative data
        var $list = $("#chapter-progress-list");
        $list.empty();
        $.each(_latestFullData.chapters_done || [], function (_, ch) {
          $list.append(
            '<li class="list-group-item done-chapter">' +
            '<i class="bi bi-check-circle-fill text-success me-2"></i>' +
            "Chapter " + escapeHtml(ch.number) + ": " + escapeHtml(ch.title) +
            "</li>"
          );
        });

        if (typeof onComplete === "function") {
          onComplete(_latestFullData);
        }
      },
      error: function () {
        // Full fetch failures are non-critical – lightweight polling continues unaffected.
        // Update the timestamp so we don't hammer the server on repeated failures.
        _lastFullFetchTime = Date.now();
        if (typeof onComplete === "function") {
          onComplete(_latestFullData);
        }
      },
    });
  }

  function pollProgress() {
    if (!_progressToken) return;

    $.ajax({
      url: "/progress/" + _progressToken,
      method: "GET",
      success: function (data) {
        // Reset failure counter and clear any connection-lost warning
        if (_pollFailures >= 5) {
          clearAlerts();
        }
        _pollFailures = 0;

        var current = data.current || 0;
        var total = data.total || _totalChapters;
        var step = data.step || "";

        updateProgressBar(current, total, step || null);

        // Track chapter completion times for ETA estimation
        var chapterJustCompleted = current > _lastCompletedCount && current <= total;
        if (chapterJustCompleted) {
          var now = Date.now();
          // Record one timestamp per newly completed chapter
          for (var c = _lastCompletedCount; c < current; c++) {
            _chapterCompletionTimes.push(now);
          }
          _lastCompletedCount = current;
        }
        _updateTimeEstimate(current, total);

        // Adaptive backoff: reset delay when progress changes, else double it
        if (step !== _lastPollStep) {
          _pollDelay = _pollDelayMin;
        } else {
          _pollDelay = Math.min(_pollDelay * 2, _pollDelayMax);
        }
        _lastPollStep = step;

        if (data.status === "done") {
          // Generation complete – fetch full payload before showing results
          fetchFullProgress(function (fullData) {
            showDoneStep(fullData);
          });
        } else if (data.status === "error") {
          if (data.error_code === "circuit_breaker") {
            showAlert(
              "LLM API is unavailable — 3 consecutive calls failed. " +
              "Check your API key, endpoint, and rate limits, then click " +
              "\"Approve & Write Chapters\" to retry.",
              "danger"
            );
          } else {
            showAlert("Chapter generation failed: " + (data.error || "Unknown error"));
          }
          showStep("#step-outline");
          // Re-enable the approve button so the user can retry
          $("#btn-approve-outline").prop("disabled", false);
        } else {
          // Trigger a full fetch when a chapter just completed or every 2 minutes
          var timeSinceFullFetch = Date.now() - _lastFullFetchTime;
          if (chapterJustCompleted || timeSinceFullFetch >= _fullFetchIntervalMs) {
            fetchFullProgress();
          }
          _schedulePoll();
        }
      },
      error: function () {
        _pollFailures++;
        if (_pollFailures === 5) {
          showAlert(
            "Connection lost — generation may still be running in the background. " +
            "Will keep trying to reconnect.",
            "warning"
          );
        }
        // Back off on errors too
        _pollDelay = Math.min(_pollDelay * 2, _pollDelayMax);
        _schedulePoll();
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

  function _updateTimeEstimate(current, total) {
    var $el = $("#progress-time-estimate");
    var remaining = total - current;

    // Need at least 1 completed chapter to estimate
    if (_chapterCompletionTimes.length < 1 || remaining <= 0) {
      $el.addClass("d-none").text("");
      return;
    }

    // Calculate average time per chapter from wall-clock data
    var startRef = _generationStartTime || _chapterCompletionTimes[0];
    var lastTime = _chapterCompletionTimes[_chapterCompletionTimes.length - 1];
    var elapsed = lastTime - startRef;
    var completed = _chapterCompletionTimes.length;
    var avgMs = elapsed / completed;

    var estRemainingMs = avgMs * remaining;
    var estMinutes = Math.round(estRemainingMs / 60000);

    var text;
    if (estMinutes < 1) {
      text = "Less than a minute remaining";
    } else if (estMinutes === 1) {
      text = "~1 minute remaining";
    } else if (estMinutes < 60) {
      text = "~" + estMinutes + " minutes remaining";
    } else {
      var hrs = Math.floor(estMinutes / 60);
      var mins = estMinutes % 60;
      text = "~" + hrs + "h " + mins + "m remaining";
    }

    // Also show avg time per chapter
    var avgMin = Math.round(avgMs / 60000);
    if (avgMin >= 1) {
      text += " (avg ~" + avgMin + " min/chapter)";
    }

    $el.text(text).removeClass("d-none");
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
      var $item = $(
        '<div class="accordion-item">' +
        '<h2 class="accordion-header" id="' + heading + '">' +
        '<button class="accordion-button collapsed" type="button" ' +
        'data-bs-toggle="collapse" data-bs-target="#' + id + '" ' +
        'aria-expanded="false" aria-controls="' + id + '">' +
        "Chapter " + escapeHtml(ch.number) + ": " + escapeHtml(ch.title) +
        "</button></h2>" +
        '<div id="' + id + '" class="accordion-collapse collapse" aria-labelledby="' + heading + '">' +
        '<div class="accordion-body"><pre class="mb-0 text-wrap"></pre></div>' +
        "</div></div>"
      );
      // Use .text() on the <pre> element to safely insert content with newlines preserved
      $item.find("pre").text(ch.content || "");
      $acc.append($item);
    });

    // Writing statistics dashboard
    _renderWritingStats(data.chapters_done || []);

    // Character relationship map
    _renderRelationshipMap(data.character_relationship_map);

    showStep("#step-done");
    // Generation complete — re-enable the approve button for potential future use
    $("#btn-approve-outline").prop("disabled", false);
  }

  function _renderRelationshipMap(mapData) {
    var $panel = $("#relationship-map-panel");
    var $pre = $("#relationship-mermaid");

    if (!mapData || !mapData.relationships || mapData.relationships.length === 0) {
      $panel.addClass("d-none");
      return;
    }

    // Build Mermaid flowchart definition
    var lines = ["graph LR"];
    var seen = {};

    // Assign short IDs to character names
    var chars = mapData.characters || [];
    var idMap = {};
    for (var i = 0; i < chars.length; i++) {
      var id = "C" + i;
      idMap[chars[i]] = id;
      lines.push("    " + id + '["' + chars[i].replace(/"/g, "'") + '"]');
    }

    // Add relationships as edges
    var rels = mapData.relationships || [];
    for (var j = 0; j < rels.length; j++) {
      var r = rels[j];
      var fromId = idMap[r.from];
      var toId = idMap[r.to];
      if (!fromId || !toId) continue;
      var edgeKey = fromId + "-" + toId;
      if (seen[edgeKey]) continue;
      seen[edgeKey] = true;

      var label = (r.type || "").replace(/"/g, "'");
      lines.push("    " + fromId + " -->|" + label + "| " + toId);
    }

    var mermaidCode = lines.join("\n");
    $pre.removeAttr("data-processed").text(mermaidCode);

    // Re-render with Mermaid
    try {
      if (typeof mermaid !== "undefined" && mermaid.run) {
        mermaid.run({ nodes: [$pre[0]] });
      }
    } catch (e) {
      // Fallback: show raw text
    }

    $panel.removeClass("d-none");
  }

  function _renderWritingStats(chapters) {
    var $panel = $("#writing-stats-panel");
    var $tbody = $("#writing-stats-tbody").empty();
    var $summary = $("#writing-stats-summary").empty();

    // Check if any chapter has stats (word_count field present)
    var hasStats = chapters.some(function (ch) {
      return ch.word_count || ch.generation_time_seconds || ch.total_tokens;
    });

    if (!chapters.length) {
      $panel.addClass("d-none");
      return;
    }

    // Compute per-chapter stats and totals
    var totalWords = 0;
    var totalTime = 0;
    var totalCalls = 0;
    var totalTokens = 0;

    $.each(chapters, function (_, ch) {
      var words = ch.word_count || (ch.content ? ch.content.split(/\s+/).length : 0);
      var timeSec = ch.generation_time_seconds || 0;
      var calls = ch.llm_calls || 0;
      var tokens = ch.total_tokens || 0;

      totalWords += words;
      totalTime += timeSec;
      totalCalls += calls;
      totalTokens += tokens;

      var timeStr = timeSec > 0 ? _formatDuration(timeSec) : "-";
      var tokensStr = tokens > 0 ? tokens.toLocaleString() : "-";
      var callsStr = calls > 0 ? calls.toLocaleString() : "-";

      $tbody.append(
        "<tr>" +
        "<td>" + escapeHtml(ch.number) + "</td>" +
        "<td>" + escapeHtml(ch.title || "") + "</td>" +
        '<td class="text-end">' + words.toLocaleString() + "</td>" +
        '<td class="text-end">' + timeStr + "</td>" +
        '<td class="text-end">' + callsStr + "</td>" +
        '<td class="text-end">' + tokensStr + "</td>" +
        "</tr>"
      );
    });

    // Summary cards
    var avgWords = chapters.length > 0 ? Math.round(totalWords / chapters.length) : 0;
    var summaryItems = [
      { label: "Total Words", value: totalWords.toLocaleString(), icon: "bi-file-text" },
      { label: "Avg Words/Ch", value: avgWords.toLocaleString(), icon: "bi-calculator" },
    ];
    if (totalTime > 0) {
      summaryItems.push({ label: "Total Gen Time", value: _formatDuration(totalTime), icon: "bi-clock" });
    }
    if (totalCalls > 0) {
      summaryItems.push({ label: "LLM Calls", value: totalCalls.toLocaleString(), icon: "bi-chat-dots" });
    }
    if (totalTokens > 0) {
      summaryItems.push({ label: "Total Tokens", value: totalTokens.toLocaleString(), icon: "bi-cpu" });
    }

    $.each(summaryItems, function (_, item) {
      $summary.append(
        '<div class="col-6 col-md-4 col-lg-2">' +
        '<div class="card text-center h-100">' +
        '<div class="card-body py-2 px-1">' +
        '<i class="bi ' + item.icon + ' text-primary mb-1 d-block"></i>' +
        '<div class="fw-bold">' + item.value + "</div>" +
        '<small class="text-muted">' + item.label + "</small>" +
        "</div></div></div>"
      );
    });

    $panel.removeClass("d-none");
  }

  function _formatDuration(seconds) {
    if (seconds < 60) return Math.round(seconds) + "s";
    var mins = Math.floor(seconds / 60);
    var secs = Math.round(seconds % 60);
    if (mins < 60) return mins + "m " + secs + "s";
    var hrs = Math.floor(mins / 60);
    mins = mins % 60;
    return hrs + "h " + mins + "m";
  }

  // -------------------------------------------------------------------
  // Export
  // -------------------------------------------------------------------
  // Selects all export-related buttons to disable/enable as a group
  var _$exportButtons = $("#btn-export-manuscript, #btn-export-editors-notes, #btn-generate-illustrations");

  function _disableExportButtons() {
    _$exportButtons.prop("disabled", true);
  }

  function _enableExportButtons() {
    _$exportButtons.prop("disabled", false);
  }

  $("#btn-export-manuscript").on("click", function () {
    clearAlerts();
    $("#export-manuscript-spinner").removeClass("d-none");
    _disableExportButtons();

    $.ajax({
      url: "/export",
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
        var msg = (xhr.responseJSON && xhr.responseJSON.error) || "Export failed. The novel data may be incomplete — try again.";
        showAlert(msg);
      },
      complete: function () {
        $("#export-manuscript-spinner").addClass("d-none");
        _enableExportButtons();
      },
    });
  });

  $("#btn-export-editors-notes").on("click", function () {
    clearAlerts();
    $("#export-editors-notes-spinner").removeClass("d-none");
    _disableExportButtons();

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
          "Editor's notes export failed. No diagnostic reports may be available for this novel.";
        showAlert(msg);
      },
      complete: function () {
        $("#export-editors-notes-spinner").addClass("d-none");
        _enableExportButtons();
      },
    });
  });

  // -------------------------------------------------------------------
  // Illustrations
  // -------------------------------------------------------------------
  // Render illustration cards into the gallery
  function renderIllustrations(illustrations) {
    var $gallery = $("#illustrations-gallery");
    $gallery.empty();

    if (!illustrations || illustrations.length === 0) {
      $gallery.append(
        '<div class="col-12"><p class="text-muted">No illustrations were generated.</p></div>'
      );
    } else {
      $.each(illustrations, function (i, illust) {
        var label =
          illust.type === "cover"
            ? "Cover"
            : "Chapter " + (illust.chapter || "?");
        var safeUrl = escapeHtml(illust.image_url || "");
        var $card = $(
          '<div class="col-6 col-md-4 col-lg-3">' +
            '<div class="card h-100 shadow-sm">' +
              '<a href="' + safeUrl + '" target="_blank" rel="noopener">' +
                '<img src="' + safeUrl + '" class="card-img-top" ' +
                  'alt="' + escapeHtml(label) + '" style="cursor:zoom-in;" loading="lazy">' +
              "</a>" +
              '<div class="card-body p-2">' +
                '<p class="card-title fw-bold mb-1 small"></p>' +
                '<p class="card-text text-muted small mb-0"></p>' +
              "</div>" +
            "</div>" +
          "</div>"
        );
        $card.find(".card-title").text(label);
        $card.find(".card-text").text(illust.scene_description || "");
        $gallery.append($card);
      });
    }

    $gallery.removeClass("d-none");
  }

  // Poll the illustration job token until it reaches a terminal state,
  // then fetch the full payload and render the images.
  function _pollIllustrationJob(illustToken, _errorRetries) {
    var retries = _errorRetries || 0;
    $.ajax({
      url: "/progress/" + illustToken,
      method: "GET",
      success: function (data) {
        if (data.status === "done") {
          // Job finished – fetch the full payload to get the illustrations array.
          $.ajax({
            url: "/progress/" + illustToken + "/full",
            method: "GET",
            success: function (full) {
              renderIllustrations(full.illustrations || []);
            },
            error: function () {
              renderIllustrations([]);
            },
            complete: function () {
              $("#illustrations-spinner").addClass("d-none");
              $("#illustrations-spinner-label").addClass("d-none").text("");
              _enableExportButtons();
            },
          });
        } else if (data.status === "error") {
          showAlert(
            data.error ||
              "Illustration generation failed. The AI service may be rate-limited — wait a few minutes and try again."
          );
          $("#illustrations-spinner").addClass("d-none");
          $("#illustrations-spinner-label").addClass("d-none").text("");
          _enableExportButtons();
        } else {
          // Still running – update the spinner label and keep polling.
          var stepText = data.step || "Generating\u2026";
          var current = data.current || 0;
          var total = data.total || 0;
          var label = total > 0
            ? stepText + " (" + current + "/" + total + ")"
            : stepText;
          $("#illustrations-spinner-label").text(label);
          setTimeout(function () {
            _pollIllustrationJob(illustToken, 0);
          }, 3000);
        }
      },
      error: function () {
        // Retry on transient network errors, up to 10 attempts.
        if (retries < 10) {
          setTimeout(function () {
            _pollIllustrationJob(illustToken, retries + 1);
          }, 5000);
        } else {
          showAlert(
            "Lost connection to the server while waiting for illustrations. Please check your connection and try again."
          );
          $("#illustrations-spinner").addClass("d-none");
          $("#illustrations-spinner-label").addClass("d-none").text("");
          _enableExportButtons();
        }
      },
    });
  }

  $("#btn-generate-illustrations").on("click", function () {
    clearAlerts();
    var $btn = $(this);
    $("#illustrations-spinner").removeClass("d-none");
    $("#illustrations-spinner-label").removeClass("d-none").text("Starting\u2026");
    _disableExportButtons();

    $.ajax({
      url: "/generate_illustrations",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ token: _progressToken }),
      success: function (resp) {
        var illustToken = resp.illustration_token;
        if (illustToken) {
          // Backend accepted the job; poll until done.
          _pollIllustrationJob(illustToken, 0);
        } else {
          // Unexpected: no token in response.
          renderIllustrations(resp.illustrations || []);
          $("#illustrations-spinner").addClass("d-none");
          $("#illustrations-spinner-label").addClass("d-none").text("");
          _enableExportButtons();
        }
      },
      error: function (xhr) {
        var msg =
          (xhr.responseJSON && xhr.responseJSON.error) ||
          "Illustration generation failed. The AI service may be rate-limited — wait a few minutes and try again.";
        showAlert(msg);
        $("#illustrations-spinner").addClass("d-none");
        $("#illustrations-spinner-label").addClass("d-none").text("");
        _enableExportButtons();
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
          "Chapter revision failed. The AI service may be unavailable — your original chapter is unchanged.";
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
  var _logPollTimeout = null;
  var _logPollSameCount = 0;  // consecutive polls with no new entries
  // Backoff schedule: 0-1 same polls → 15s, 2 → 30s, 3 → 60s, 4 → 2min, 5 → 5min, 6+ → 10min
  var _LOG_POLL_DELAYS = [15000, 15000, 30000, 60000, 120000, 300000, 600000];

  function _scheduleLogPoll() {
    if (_logPollTimeout) {
      clearTimeout(_logPollTimeout);
    }
    var delay = _LOG_POLL_DELAYS[Math.min(_logPollSameCount, _LOG_POLL_DELAYS.length - 1)];
    _logPollTimeout = setTimeout(pollLLMLog, delay);
  }

  function entrySignature(entry) {
    if (!entry) return "";
    var payloadMessages = (entry.payload && entry.payload.messages) || [];
    var messageCount = payloadMessages.length;
    var firstMessage = messageCount > 0 && payloadMessages[0].content ? String(payloadMessages[0].content).slice(0, 60) : "";
    return [entry.timestamp || "", entry.type || "", entry.action || "", messageCount, firstMessage].join("|");
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
          _scheduleLogPoll();
          return;
        }

        if (entries.length === 0) {
          _logPollSameCount++;
          _scheduleLogPoll();
          return;
        }

        // Clear placeholder on first visible log entry
        if (Object.keys(_seenLogSignatures).length === 0) {
          $("#llm-chat-messages").empty();
        }

        var foundNew = false;
        for (var i = 0; i < entries.length; i++) {
          var entry = entries[i];
          var signature = entrySignature(entry);
          if (!signature || _seenLogSignatures[signature]) {
            continue;
          }

          foundNew = true;
          _seenLogSignatures[signature] = true;

          if (entry.type === "request") {
            _activeLLMRequests += 1;
            var statusText = entry.action || inferStatusFromRequestEntry(entry);
            setStickyStatus(statusText);
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

        if (foundNew) {
          _logPollSameCount = 0;
        } else {
          _logPollSameCount++;
        }
        _scheduleLogPoll();
      },
      error: function() {
        // Silently fail - log polling is non-critical
        _logPollSameCount++;
        _scheduleLogPoll();
      }
    });
  }

  // Start polling for LLM log updates (with adaptive backoff)
  pollLLMLog(); // Initial poll; subsequent polls are scheduled inside pollLLMLog

  // Clear log button — clears display and server-side log file
  $("#btn-clear-log").on("click", function() {
    $("#llm-chat-messages").html(
      '<div class="text-center text-muted small py-3">' +
      '<i class="bi bi-info-circle me-1"></i>Log cleared (still polling)' +
      '</div>'
    );
    _seenLogSignatures = {};
    _activeLLMRequests = 0;
    _hasInitializedLogSnapshot = false;
    _logPollSameCount = 0;
    setStickyStatus(DEFAULT_STICKY_STATUS, { force: true });

    // Clear the server-side log file
    $.post("/clear_log").fail(function () {
      // Non-fatal — display was already cleared
    });
  });

  // -------------------------------------------------------------------
  // Start Over
  // -------------------------------------------------------------------
  $("#btn-start-over").on("click", function () {
    clearAlerts();
    clearTimeout(_pollInterval);
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

  // -------------------------------------------------------------------
  // Restore session state on page load (if data was injected by Flask)
  // -------------------------------------------------------------------
  if (window._savedSessionData) {
    (function () {
      var sd = window._savedSessionData;

      // Step 1 form fields
      if (sd.premise) $("#premise").val(sd.premise);
      if (sd.genre) $("#genre").val(sd.genre);
      if (sd.chapters) $("#chapters").val(sd.chapters);
      if (sd.word_count) $("#word_count").val(sd.word_count);
      if (sd.special_events) $("#special_events").val(sd.special_events);
      if (sd.special_instructions) $("#special_instructions").val(sd.special_instructions);
      $("#premise-count").text((sd.premise || "").length);

      // Step 2 outline + characters
      if (sd.chapter_list && sd.chapter_list.length) {
        renderOutline({
          title: sd.title || "",
          chapters: sd.chapter_list,
          characters: sd.character_list || [],
        });
        // Restore narrative perspective after characters are rendered
        if (sd.narrative_perspective) {
          $("#narrative-perspective").val(sd.narrative_perspective);
        }
      }

      // Step 4 completion data — restore progress token and show done step
      var pd = sd.progress_data;
      var cc = sd.completed_chapters;
      if (sd.progress_token && pd) {
        _progressToken = sd.progress_token;

        if (pd.status === "done") {
          showDoneStep(pd);
        } else if (pd.status === "running") {
          // Generation still in progress — show progress tab and start polling
          _totalChapters = sd.chapters || 20;
          showStep("#step-progress");
          _pollDelay = _pollDelayMin;
          _lastPollStep = "";
          _pollFailures = 0;
          _lastCompletedCount = pd.current || 0;
          _chapterCompletionTimes = [];
          _generationStartTime = Date.now();
          _schedulePoll();
          pollProgress();
        } else if (pd.chapters_done && pd.chapters_done.length) {
          // Has some chapters (e.g. errored mid-run) — show what we have
          showDoneStep(pd);
        } else {
          // Has outline but no generation yet — show outline tab
          showStep("#step-outline");
        }
      } else if (cc && cc.length) {
        // No progress_data in memory but completed chapters saved in session file
        // (e.g. after server restart) — rebuild and show done step
        _progressToken = sd.progress_token || "";
        showDoneStep({
          status: "done",
          chapters_done: cc,
          current: cc.length,
          total: sd.chapters || cc.length,
        });
      } else if (sd.chapter_list && sd.chapter_list.length) {
        // Has outline but no progress — show outline tab
        showStep("#step-outline");
      }

      // Restore saved illustrations if available
      if (sd.illustrations && sd.illustrations.length) {
        renderIllustrations(sd.illustrations);
      }

      delete window._savedSessionData;
    })();
  }
});
