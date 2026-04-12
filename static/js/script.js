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

  // Map step IDs to their index in the step indicator (0-based)
  var STEP_INDEX = {
    "#step-input": 0,
    "#step-outline": 1,
    "#step-progress": 2,
    "#step-done": 3,
  };

  function _updateStepIndicator(activeId) {
    // NOTE: We deliberately do NOT touch aria-selected here. Bootstrap Tab
    // uses aria-selected to find the previously active tab when deactivating
    // it; if we strip it preemptively, Bootstrap fails to deactivate the old
    // tab-pane and both panes end up visible at once.
    var activeIdx = STEP_INDEX[activeId];
    if (activeIdx === undefined) {
      // Log tab or unknown — visually dim step circles but preserve the
      // Bootstrap-managed .active class so Tab.show() can still find
      // and deactivate the previous pane.
      $(".nf-step").removeClass("completed").addClass("text-muted");
      $(".nf-step-line").removeClass("completed");
      $(".nf-step-log-btn").addClass("active");
      return;
    }
    $(".nf-step-log-btn").removeClass("active");
    $(".nf-step").each(function (i) {
      var $step = $(this);
      $step.removeClass("active completed text-muted");
      if (i < activeIdx) {
        $step.addClass("completed");
        // Replace number with checkmark for completed steps
        $step.find(".nf-step-circle").html('<i class="bi bi-check"></i>');
      } else if (i === activeIdx) {
        $step.addClass("active");
        $step.find(".nf-step-circle").text(i + 1);
      } else {
        $step.find(".nf-step-circle").text(i + 1);
      }
    });
    $(".nf-step-line").each(function (i) {
      $(this).toggleClass("completed", i < activeIdx);
    });
  }

  function showStep(id) {
    var tabButtonSelector = STEP_TAB_BUTTONS[id];
    var tabButton = tabButtonSelector ? document.querySelector(tabButtonSelector) : null;
    if (tabButton) {
      bootstrap.Tab.getOrCreateInstance(tabButton).show();
    }

    // Update the step progress indicator
    _updateStepIndicator(id);

    // Hide the welcome hero once user navigates away from initial state
    $("#nf-hero").addClass("d-none");

    // Legacy cleanup: ensure old section-based d-none classes do not hide content.
    $.each(STEPS, function (_, sel) {
      $(sel).removeClass("d-none");
    });
    $("html, body").animate({ scrollTop: 0 }, 200);
  }

  // Sync step indicator when user clicks step buttons directly
  $(document).on("shown.bs.tab", ".nf-step, .nf-step-log-btn", function () {
    var id = null;
    var target = $(this).attr("data-bs-target");
    // Find which step ID maps to this tab target
    if (target === "#step1-novel-setup-tab") id = "#step-input";
    else if (target === "#step2-plan-tab") id = "#step-outline";
    else if (target === "#step3-chapter-writing-tab") id = "#step-progress";
    else if (target === "#step4-complete-export-tab") id = "#step-done";
    else if (target === "#log-tab") { _updateStepIndicator(null); return; }
    if (id) _updateStepIndicator(id);
  });

  // -------------------------------------------------------------------
  // Alert helpers
  // -------------------------------------------------------------------
  function showAlert(message, type) {
    type = type || "danger";
    var icons = {
      danger: "bi-exclamation-triangle-fill",
      warning: "bi-exclamation-circle-fill",
      info: "bi-info-circle-fill",
      success: "bi-check-circle-fill",
    };
    var icon = icons[type] || icons.danger;
    var html =
      '<div class="nf-toast nf-toast-' + type + ' alert alert-' + type +
      ' alert-dismissible fade show" role="alert">' +
      '<i class="bi ' + icon + ' me-2"></i>' +
      escapeHtml(message) +
      '<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>' +
      '<div class="nf-toast-progress"></div>' +
      "</div>";
    var $area = $("#global-alert-area").html(html);
    $("html, body").animate({ scrollTop: 0 }, 200);

    // Auto-dismiss after 8 seconds
    var $toast = $area.find(".nf-toast");
    var timer = setTimeout(function () {
      $toast.alert("close");
    }, 8000);
    // Cancel auto-dismiss if user closes manually
    $toast.on("close.bs.alert", function () {
      clearTimeout(timer);
    });
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
    if (combined.indexOf("voice & dialogue") !== -1 || combined.indexOf("voice and dialogue") !== -1) {
      return "Differentiating character voices";
    }
    if (combined.indexOf("human oddities") !== -1) {
      return "Adding human texture";
    }
    if (combined.indexOf("metaphor reduction") !== -1) {
      return "Reducing metaphor density";
    }
    if (combined.indexOf("copy edit") !== -1) {
      return "Copy editing";
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
        $menu.append('<li><span class="dropdown-item text-muted"><i class="bi bi-journal me-2"></i>No saved stories yet</span></li>');
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

    // Show spinner and update button text
    $("#outline-spinner").removeClass("d-none");
    $("#outline-btn-icon").removeClass("bi-magic").addClass("bi-stars");
    $("#outline-btn-text").text("Conjuring your story\u2026");
    $("#btn-generate-outline").prop("disabled", true);

    // Show skeleton placeholders in Step 2 and switch to it immediately
    _showOutlineSkeletons(chapters);
    showStep("#step-outline");

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
        _hideOutlineSkeletons();
        renderOutline(resp);
      },
      error: function (xhr) {
        _hideOutlineSkeletons();
        var msg =
          (xhr.responseJSON && xhr.responseJSON.error) ||
          "Failed to generate outline. Check your LLM API configuration.";
        showAlert(msg);
        showStep("#step-input");
      },
      complete: function () {
        $("#outline-spinner").addClass("d-none");
        $("#outline-btn-icon").removeClass("bi-stars bi-check-lg").addClass("bi-magic");
        $("#outline-btn-text").text("Generate Outline");
        $("#btn-generate-outline").prop("disabled", false);
      },
    });
  });

  // -------------------------------------------------------------------
  // Skeleton loading screens for outline generation
  // -------------------------------------------------------------------
  function _showOutlineSkeletons(chapterCount) {
    // Chapter skeletons in card view
    var $cards = $("#chapter-cards").empty();
    for (var i = 0; i < chapterCount; i++) {
      $cards.append(
        '<div class="nf-chapter-card nf-skeleton-card">' +
        '<div class="nf-chapter-card-header">' +
        '<span class="nf-chapter-badge nf-skeleton-pulse">&nbsp;</span>' +
        '<div class="nf-skeleton-bar nf-skeleton-pulse" style="width:40%;height:1.1rem"></div>' +
        '</div>' +
        '<div class="nf-skeleton-bar nf-skeleton-pulse" style="width:90%;height:0.85rem;margin-top:0.5rem"></div>' +
        '<div class="nf-skeleton-bar nf-skeleton-pulse" style="width:70%;height:0.85rem;margin-top:0.4rem"></div>' +
        '</div>'
      );
    }

    // Character skeletons in card grid
    var $charCards = $("#character-cards");
    $charCards.find(".col-md-6:not(#add-character-card-col)").remove();
    for (var j = 0; j < 4; j++) {
      var colorClass = "nf-char-color-" + (j % 8);
      $(
        '<div class="col-md-6">' +
        '<div class="nf-char-card ' + colorClass + ' nf-skeleton-card">' +
        '<div class="nf-skeleton-bar nf-skeleton-pulse" style="width:50%;height:1.2rem;margin-bottom:0.5rem"></div>' +
        '<div class="nf-skeleton-bar nf-skeleton-pulse" style="width:30%;height:0.9rem;margin-bottom:0.75rem;border-radius:1rem"></div>' +
        '<div class="nf-skeleton-bar nf-skeleton-pulse" style="width:80%;height:0.75rem;margin-bottom:0.3rem"></div>' +
        '<div class="nf-skeleton-bar nf-skeleton-pulse" style="width:60%;height:0.75rem"></div>' +
        '</div>' +
        '</div>'
      ).insertBefore("#add-character-card-col");
    }

    // Disable the approve button while loading
    $("#btn-approve-outline").prop("disabled", true);
  }

  function _hideOutlineSkeletons() {
    $(".nf-skeleton-card").remove();
    $("#btn-approve-outline").prop("disabled", false);
  }

  // -------------------------------------------------------------------
  // Render outline into the review table
  // -------------------------------------------------------------------
  function renderOutline(data) {
    $("#outline-title").val(data.title || "");

    // Chapters — build both card and table views
    var $tbody = $("#chapter-tbody").empty();
    var $cards = $("#chapter-cards").empty();
    // Re-add the "add character" card placeholder (it was emptied above only for chapters)
    _charColorIdx = 0;
    $.each(data.chapters || [], function (_, ch) {
      addChapterRow(ch.number || "", ch.title || "", ch.summary || "");
    });

    // Characters — clear cards (except add button) and hidden table
    $("#character-cards .col-md-6:not(#add-character-card-col)").remove();
    var $ctbody = $("#characters-tbody").empty();
    _charColorIdx = 0;
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
  var _charColorIdx = 0;

  function addCharacterRow(name, age, role, background, arc) {
    var safeName = escapeHtml(name);
    var colorClass = "nf-char-color-" + (_charColorIdx++ % 8);

    // Hidden table row (used for data collection on approve)
    var row =
      "<tr>" +
      "<td><div class='editable-cell' data-field='name'>" + safeName + "</div></td>" +
      "<td><div class='editable-cell' data-field='age'>" + escapeHtml(age) + "</div></td>" +
      "<td><div class='editable-cell' data-field='role'>" + escapeHtml(role) + "</div></td>" +
      "<td><div class='editable-cell' data-field='background'>" + escapeHtml(background) + "</div></td>" +
      "<td><div class='editable-cell' data-field='arc'>" + escapeHtml(arc) + "</div></td>" +
      "</tr>";
    $("#characters-tbody").append(row);

    // Unique IDs for collapse sections
    var uid = "char-" + Date.now() + "-" + Math.random().toString(36).substr(2, 5);

    // Card
    var card =
      '<div class="col-md-6">' +
      '<div class="nf-char-card ' + colorClass + '">' +
      '<button class="btn btn-sm btn-outline-danger nf-char-card-delete" title="Delete Character" aria-label="Delete ' + safeName + '"><i class="bi bi-trash"></i></button>' +
      '<div class="nf-char-card-name editable-cell" contenteditable="true" data-field="name" role="textbox" aria-label="Character name">' + safeName + '</div>' +
      '<div class="d-flex align-items-center gap-2 mb-2">' +
      '<span class="nf-char-card-role editable-cell" contenteditable="true" data-field="role" role="textbox" aria-label="Role">' + escapeHtml(role) + '</span>' +
      '<span class="nf-char-card-age">Age: <span class="editable-cell" contenteditable="true" data-field="age" role="textbox" aria-label="Age">' + escapeHtml(age) + '</span></span>' +
      '</div>' +
      '<div class="mb-2">' +
      '<div class="nf-char-card-section-label" data-bs-toggle="collapse" data-bs-target="#bg-' + uid + '" aria-expanded="false"><i class="bi bi-chevron-right me-1"></i>Background</div>' +
      '<div class="collapse" id="bg-' + uid + '">' +
      '<div class="nf-char-card-body editable-cell mt-1" contenteditable="true" data-field="background" role="textbox" aria-label="Background">' + escapeHtml(background) + '</div>' +
      '</div>' +
      '</div>' +
      '<div>' +
      '<div class="nf-char-card-section-label" data-bs-toggle="collapse" data-bs-target="#arc-' + uid + '" aria-expanded="false"><i class="bi bi-chevron-right me-1"></i>Arc</div>' +
      '<div class="collapse" id="arc-' + uid + '">' +
      '<div class="nf-char-card-body editable-cell mt-1" contenteditable="true" data-field="arc" role="textbox" aria-label="Arc">' + escapeHtml(arc) + '</div>' +
      '</div>' +
      '</div>' +
      '</div>' +
      '</div>';
    // Insert before the add-character card
    $(card).insertBefore("#add-character-card-col");
  }

  // Sync hidden table from character cards (called before approve)
  function _syncCharTableFromCards() {
    var $tbody = $("#characters-tbody").empty();
    $("#character-cards .nf-char-card:not(.nf-char-card-add)").each(function () {
      var $card = $(this);
      var row =
        "<tr>" +
        "<td><div class='editable-cell' data-field='name'>" + escapeHtml($card.find("[data-field='name']").text().trim()) + "</div></td>" +
        "<td><div class='editable-cell' data-field='age'>" + escapeHtml($card.find("[data-field='age']").text().trim()) + "</div></td>" +
        "<td><div class='editable-cell' data-field='role'>" + escapeHtml($card.find("[data-field='role']").text().trim()) + "</div></td>" +
        "<td><div class='editable-cell' data-field='background'>" + escapeHtml($card.find("[data-field='background']").text().trim()) + "</div></td>" +
        "<td><div class='editable-cell' data-field='arc'>" + escapeHtml($card.find("[data-field='arc']").text().trim()) + "</div></td>" +
        "</tr>";
      $tbody.append(row);
    });
  }

  // Sync the Narrative Perspective dropdown with current character names
  function syncPerspectiveDropdown() {
    var $select = $("#narrative-perspective");
    var currentVal = $select.val();
    // Remove all first-person options (keep third_person)
    $select.find("option[value!='third_person']").remove();
    // Add an option for each character from cards
    $("#character-cards .nf-char-card:not(.nf-char-card-add)").each(function () {
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

  // Add Character button (card-based)
  $(document).on("click", "#btn-add-character", function () {
    addCharacterRow("New Character", "", "Protagonist/Antagonist/Supporting", "Enter background...", "Enter character arc...");
    markOutlineDirty();
    syncPerspectiveDropdown();
  });

  // Delete Character button from card
  $("#character-cards").on("click", ".nf-char-card-delete", function () {
    var $col = $(this).closest(".col-md-6");
    var characterName = $col.find("[data-field='name']").text().trim();

    if ($("#character-cards .nf-char-card:not(.nf-char-card-add)").length <= 1) {
      showAlert("Cannot delete the last character. At least one character is required.", "warning");
      return;
    }

    if (confirm("Delete character '" + characterName + "'?")) {
      $col.remove();
      _syncCharTableFromCards();
      markOutlineDirty();
      syncPerspectiveDropdown();
    }
  });

  // Mark dirty on character card edits
  $(document).on("input", "#character-cards .editable-cell", markOutlineDirty);

  // -------------------------------------------------------------------
  // Add/Delete Chapter Functions
  // -------------------------------------------------------------------
  function addChapterRow(number, title, summary) {
    var chLabel = "Chapter " + escapeHtml(number);

    // Table row
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

    // Card
    var card =
      '<div class="nf-chapter-card" draggable="true">' +
      '<div class="nf-chapter-kebab dropdown">' +
      '<button class="btn btn-sm btn-link text-muted" data-bs-toggle="dropdown" aria-expanded="false" title="Actions"><i class="bi bi-three-dots-vertical"></i></button>' +
      '<ul class="dropdown-menu dropdown-menu-end">' +
      '<li><button class="dropdown-item btn-card-move-up"><i class="bi bi-arrow-up me-2"></i>Move Up</button></li>' +
      '<li><button class="dropdown-item btn-card-move-down"><i class="bi bi-arrow-down me-2"></i>Move Down</button></li>' +
      '<li><hr class="dropdown-divider"></li>' +
      '<li><button class="dropdown-item btn-card-add-before"><i class="bi bi-plus-circle me-2"></i>Add Before</button></li>' +
      '<li><button class="dropdown-item btn-card-add-after"><i class="bi bi-plus-circle me-2"></i>Add After</button></li>' +
      '<li><hr class="dropdown-divider"></li>' +
      '<li><button class="dropdown-item text-danger btn-card-delete"><i class="bi bi-trash me-2"></i>Delete</button></li>' +
      '</ul>' +
      '</div>' +
      '<div class="nf-chapter-card-header">' +
      '<span class="nf-chapter-badge chapter-number">' + escapeHtml(number) + '</span>' +
      '<div class="nf-chapter-card-title editable-cell" contenteditable="true" data-field="title" role="textbox" aria-label="' + chLabel + ' title">' + escapeHtml(title) + '</div>' +
      '</div>' +
      '<div class="nf-chapter-card-summary editable-cell" contenteditable="true" data-field="summary" role="textbox" aria-label="' + chLabel + ' summary">' + escapeHtml(summary) + '</div>' +
      '</div>';
    $("#chapter-cards").append(card);

    renumberChapters();
  }

  function renumberChapters() {
    $("#chapter-tbody tr").each(function (idx) {
      $(this).find(".chapter-number").text(idx + 1);
    });
    $("#chapter-cards .nf-chapter-card").each(function (idx) {
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

  function buildNewChapterRowHtml() {
    return (
      "<tr>" +
      "<td class='chapter-number'></td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='title' role='textbox' aria-label='Chapter title'>New Chapter</div></td>" +
      "<td><div class='editable-cell' contenteditable='true' data-field='summary' role='textbox' aria-label='Chapter summary'>Enter chapter summary...</div></td>" +
      "<td class='text-center'>" +
      "<div class='btn-group btn-group-sm me-1' role='group' aria-label='Reorder chapter'>" +
      "<button class='btn btn-outline-secondary btn-move-up' title='Move Up' aria-label='Move chapter up'><i class='bi bi-arrow-up'></i></button>" +
      "<button class='btn btn-outline-secondary btn-move-down' title='Move Down' aria-label='Move chapter down'><i class='bi bi-arrow-down'></i></button>" +
      "</div>" +
      "<div class='btn-group btn-group-sm me-1' role='group' aria-label='Insert around chapter'>" +
      "<button class='btn btn-outline-success btn-add-before' title='Add Before' aria-label='Add chapter before'><i class='bi bi-plus-circle'></i></button>" +
      "<button class='btn btn-outline-success btn-add-after' title='Add After' aria-label='Add chapter after'><i class='bi bi-plus-circle'></i></button>" +
      "</div>" +
      "<button class='btn btn-sm btn-outline-danger btn-delete-chapter' title='Delete Chapter' aria-label='Delete chapter'><i class='bi bi-trash'></i></button>" +
      "</td>" +
      "</tr>"
    );
  }

  // Add chapter before
  $("#chapter-tbody").on("click", ".btn-add-before", function () {
    var $row = $(this).closest("tr");
    $row.before(buildNewChapterRowHtml());
    renumberChapters();
    markOutlineDirty();
  });

  // Add chapter after
  $("#chapter-tbody").on("click", ".btn-add-after", function () {
    var $row = $(this).closest("tr");
    $row.after(buildNewChapterRowHtml());
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
  // Card view helpers and event handlers
  // -------------------------------------------------------------------
  function buildNewChapterCardHtml() {
    return (
      '<div class="nf-chapter-card" draggable="true">' +
      '<div class="nf-chapter-kebab dropdown">' +
      '<button class="btn btn-sm btn-link text-muted" data-bs-toggle="dropdown" aria-expanded="false" title="Actions"><i class="bi bi-three-dots-vertical"></i></button>' +
      '<ul class="dropdown-menu dropdown-menu-end">' +
      '<li><button class="dropdown-item btn-card-move-up"><i class="bi bi-arrow-up me-2"></i>Move Up</button></li>' +
      '<li><button class="dropdown-item btn-card-move-down"><i class="bi bi-arrow-down me-2"></i>Move Down</button></li>' +
      '<li><hr class="dropdown-divider"></li>' +
      '<li><button class="dropdown-item btn-card-add-before"><i class="bi bi-plus-circle me-2"></i>Add Before</button></li>' +
      '<li><button class="dropdown-item btn-card-add-after"><i class="bi bi-plus-circle me-2"></i>Add After</button></li>' +
      '<li><hr class="dropdown-divider"></li>' +
      '<li><button class="dropdown-item text-danger btn-card-delete"><i class="bi bi-trash me-2"></i>Delete</button></li>' +
      '</ul>' +
      '</div>' +
      '<div class="nf-chapter-card-header">' +
      '<span class="nf-chapter-badge chapter-number"></span>' +
      '<div class="nf-chapter-card-title editable-cell" contenteditable="true" data-field="title" role="textbox" aria-label="Chapter title">New Chapter</div>' +
      '</div>' +
      '<div class="nf-chapter-card-summary editable-cell" contenteditable="true" data-field="summary" role="textbox" aria-label="Chapter summary">Enter chapter summary...</div>' +
      '</div>'
    );
  }

  // Sync: rebuild table rows from cards (called when switching to table view or on approve)
  function _syncTableFromCards() {
    var $tbody = $("#chapter-tbody").empty();
    $("#chapter-cards .nf-chapter-card").each(function (idx) {
      var $card = $(this);
      var title = $card.find("[data-field='title']").text().trim();
      var summary = $card.find("[data-field='summary']").text().trim();
      // Build table row without re-calling addChapterRow (avoids recursion)
      var chLabel = "Chapter " + (idx + 1);
      var row =
        "<tr>" +
        "<td class='chapter-number'>" + (idx + 1) + "</td>" +
        "<td><div class='editable-cell' contenteditable='true' data-field='title' role='textbox' aria-label='" + chLabel + " title'>" +
        escapeHtml(title) + "</div></td>" +
        "<td><div class='editable-cell' contenteditable='true' data-field='summary' role='textbox' aria-label='" + chLabel + " summary'>" +
        escapeHtml(summary) + "</div></td>" +
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
        "</td></tr>";
      $tbody.append(row);
    });
  }

  // Sync: rebuild cards from table rows (called when switching to card view)
  function _syncCardsFromTable() {
    var $cards = $("#chapter-cards").empty();
    $("#chapter-tbody tr").each(function (idx) {
      var $row = $(this);
      var title = $row.find("[data-field='title']").text().trim();
      var summary = $row.find("[data-field='summary']").text().trim();
      var num = idx + 1;
      var chLabel = "Chapter " + num;
      var card =
        '<div class="nf-chapter-card" draggable="true">' +
        '<div class="nf-chapter-kebab dropdown">' +
        '<button class="btn btn-sm btn-link text-muted" data-bs-toggle="dropdown" aria-expanded="false" title="Actions"><i class="bi bi-three-dots-vertical"></i></button>' +
        '<ul class="dropdown-menu dropdown-menu-end">' +
        '<li><button class="dropdown-item btn-card-move-up"><i class="bi bi-arrow-up me-2"></i>Move Up</button></li>' +
        '<li><button class="dropdown-item btn-card-move-down"><i class="bi bi-arrow-down me-2"></i>Move Down</button></li>' +
        '<li><hr class="dropdown-divider"></li>' +
        '<li><button class="dropdown-item btn-card-add-before"><i class="bi bi-plus-circle me-2"></i>Add Before</button></li>' +
        '<li><button class="dropdown-item btn-card-add-after"><i class="bi bi-plus-circle me-2"></i>Add After</button></li>' +
        '<li><hr class="dropdown-divider"></li>' +
        '<li><button class="dropdown-item text-danger btn-card-delete"><i class="bi bi-trash me-2"></i>Delete</button></li>' +
        '</ul></div>' +
        '<div class="nf-chapter-card-header">' +
        '<span class="nf-chapter-badge chapter-number">' + num + '</span>' +
        '<div class="nf-chapter-card-title editable-cell" contenteditable="true" data-field="title" role="textbox" aria-label="' + chLabel + ' title">' + escapeHtml(title) + '</div>' +
        '</div>' +
        '<div class="nf-chapter-card-summary editable-cell" contenteditable="true" data-field="summary" role="textbox" aria-label="' + chLabel + ' summary">' + escapeHtml(summary) + '</div>' +
        '</div>';
      $cards.append(card);
    });
  }

  // Card view: move up
  $("#chapter-cards").on("click", ".btn-card-move-up", function () {
    var $card = $(this).closest(".nf-chapter-card");
    var $prev = $card.prev(".nf-chapter-card");
    if ($prev.length) {
      $card.insertBefore($prev);
      renumberChapters();
      _syncTableFromCards();
      markOutlineDirty();
    }
  });

  // Card view: move down
  $("#chapter-cards").on("click", ".btn-card-move-down", function () {
    var $card = $(this).closest(".nf-chapter-card");
    var $next = $card.next(".nf-chapter-card");
    if ($next.length) {
      $card.insertAfter($next);
      renumberChapters();
      _syncTableFromCards();
      markOutlineDirty();
    }
  });

  // Card view: add before
  $("#chapter-cards").on("click", ".btn-card-add-before", function () {
    var $card = $(this).closest(".nf-chapter-card");
    $card.before(buildNewChapterCardHtml());
    renumberChapters();
    _syncTableFromCards();
    markOutlineDirty();
  });

  // Card view: add after
  $("#chapter-cards").on("click", ".btn-card-add-after", function () {
    var $card = $(this).closest(".nf-chapter-card");
    $card.after(buildNewChapterCardHtml());
    renumberChapters();
    _syncTableFromCards();
    markOutlineDirty();
  });

  // Card view: delete
  $("#chapter-cards").on("click", ".btn-card-delete", function () {
    var $card = $(this).closest(".nf-chapter-card");
    var chapterNum = $card.find(".chapter-number").text();
    if ($("#chapter-cards .nf-chapter-card").length <= 1) {
      showAlert("Cannot delete the last chapter. At least one chapter is required.", "warning");
      return;
    }
    if (confirm("Delete Chapter " + chapterNum + "?")) {
      $card.remove();
      renumberChapters();
      _syncTableFromCards();
      markOutlineDirty();
    }
  });

  // Card view: mark dirty on edit
  $(document).on("input", "#chapter-cards .editable-cell", markOutlineDirty);

  // -------------------------------------------------------------------
  // View toggle (Card / Table)
  // -------------------------------------------------------------------
  $("#btn-view-cards").on("click", function () {
    if ($(this).hasClass("active")) return;
    $(this).addClass("active");
    $("#btn-view-table").removeClass("active");
    _syncCardsFromTable();
    $("#chapter-cards").removeClass("d-none");
    $("#chapter-table-wrap").addClass("d-none");
  });

  $("#btn-view-table").on("click", function () {
    if ($(this).hasClass("active")) return;
    $(this).addClass("active");
    $("#btn-view-cards").removeClass("active");
    _syncTableFromCards();
    $("#chapter-table-wrap").removeClass("d-none");
    $("#chapter-cards").addClass("d-none");
  });

  // -------------------------------------------------------------------
  // Drag and drop for chapter cards
  // -------------------------------------------------------------------
  var _draggedCard = null;

  $("#chapter-cards").on("dragstart", ".nf-chapter-card", function (e) {
    _draggedCard = this;
    $(this).addClass("dragging");
    e.originalEvent.dataTransfer.effectAllowed = "move";
  });

  $("#chapter-cards").on("dragend", ".nf-chapter-card", function () {
    $(this).removeClass("dragging");
    $(".nf-chapter-card").removeClass("drag-over");
    _draggedCard = null;
  });

  $("#chapter-cards").on("dragover", ".nf-chapter-card", function (e) {
    e.preventDefault();
    e.originalEvent.dataTransfer.dropEffect = "move";
    if (this !== _draggedCard) {
      $(this).addClass("drag-over");
    }
  });

  $("#chapter-cards").on("dragleave", ".nf-chapter-card", function () {
    $(this).removeClass("drag-over");
  });

  $("#chapter-cards").on("drop", ".nf-chapter-card", function (e) {
    e.preventDefault();
    $(this).removeClass("drag-over");
    if (_draggedCard && this !== _draggedCard) {
      var $dragged = $(_draggedCard);
      var $target = $(this);
      // Determine if we should insert before or after based on position
      var targetRect = this.getBoundingClientRect();
      var midY = targetRect.top + targetRect.height / 2;
      if (e.originalEvent.clientY < midY) {
        $dragged.insertBefore($target);
      } else {
        $dragged.insertAfter($target);
      }
      renumberChapters();
      _syncTableFromCards();
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

    // Sync table from cards if card view is active
    if (!$("#chapter-cards").hasClass("d-none")) {
      _syncTableFromCards();
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

    // Sync character table from cards
    _syncCharTableFromCards();

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
    showStep("#step-progress");

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
        _rebuildTimeline([], _totalChapters);
        _startElapsedTimer();
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

        // Refresh the chapter timeline with authoritative data
        _rebuildTimeline(_latestFullData.chapters_done || [], _latestFullData.total || _totalChapters);

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
        _updateTimelineStep(step);

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

  // Build the chapter progress timeline from completed chapters + upcoming slots
  function _rebuildTimeline(chaptersDone, total) {
    var $list = $("#chapter-progress-list").empty();
    var doneCount = chaptersDone.length;

    // Completed chapters
    $.each(chaptersDone, function (_, ch) {
      var uid = "tl-preview-" + ch.number;
      var node =
        '<div class="nf-timeline-node done">' +
        '<div class="nf-timeline-dot"><i class="bi bi-check"></i></div>' +
        '<div class="nf-timeline-title">Chapter ' + escapeHtml(ch.number) + ': ' + escapeHtml(ch.title) + '</div>' +
        '<button class="nf-timeline-expand" data-bs-toggle="collapse" data-bs-target="#' + uid + '" aria-expanded="false">' +
        '<i class="bi bi-chevron-right me-1"></i>Preview</button>' +
        '<div class="collapse" id="' + uid + '"><div class="nf-timeline-preview"></div></div>' +
        '</div>';
      var $node = $(node);
      // Safely set content text (preserves newlines via pre-wrap CSS)
      $node.find(".nf-timeline-preview").text(ch.content || "");
      $list.append($node);
    });

    // Update word counter
    var totalWords = 0;
    $.each(chaptersDone, function (_, ch) {
      totalWords += ch.word_count || (ch.content ? ch.content.split(/\s+/).length : 0);
    });
    if (totalWords > 0) {
      $("#writing-word-count").text(totalWords.toLocaleString());
      $("#writing-word-counter").removeClass("d-none");
    }

    // In-progress chapter (if generation is still running)
    if (doneCount < total) {
      var inProgressNum = doneCount + 1;
      // Find chapter title from outline if available
      var inProgressTitle = "";
      var $outlineCards = $("#chapter-cards .nf-chapter-card");
      if ($outlineCards.length >= inProgressNum) {
        inProgressTitle = $outlineCards.eq(inProgressNum - 1).find("[data-field='title']").text().trim();
      }
      var titleDisplay = inProgressTitle ? "Chapter " + inProgressNum + ": " + escapeHtml(inProgressTitle) : "Chapter " + inProgressNum;

      // Update callout
      $("#writing-callout-title").text(inProgressTitle || ("Chapter " + inProgressNum));
      $("#writing-callout").removeClass("d-none");

      $list.append(
        '<div class="nf-timeline-node in-progress" id="tl-in-progress">' +
        '<div class="nf-timeline-dot"><i class="bi bi-pen"></i></div>' +
        '<div class="nf-timeline-title">' + titleDisplay + '</div>' +
        '<div class="nf-timeline-step" id="tl-step-label"></div>' +
        '</div>'
      );

      // Upcoming chapters
      for (var i = inProgressNum + 1; i <= total; i++) {
        var upTitle = "";
        if ($outlineCards.length >= i) {
          upTitle = $outlineCards.eq(i - 1).find("[data-field='title']").text().trim();
        }
        var upDisplay = upTitle ? "Chapter " + i + ": " + escapeHtml(upTitle) : "Chapter " + i;
        $list.append(
          '<div class="nf-timeline-node upcoming">' +
          '<div class="nf-timeline-dot"></div>' +
          '<div class="nf-timeline-title">' + upDisplay + '</div>' +
          '</div>'
        );
      }
    } else {
      // Generation complete — hide callout
      $("#writing-callout").addClass("d-none");
    }
  }

  // Update the in-progress node's step label (called from lightweight poll)
  function _updateTimelineStep(step) {
    var $label = $("#tl-step-label");
    if ($label.length && step) {
      $label.text(step);
    }
  }

  var _elapsedTimer = null;

  function _startElapsedTimer() {
    if (_elapsedTimer) return;
    _updateElapsed();
    _elapsedTimer = setInterval(_updateElapsed, 30000); // every 30s
  }

  function _stopElapsedTimer() {
    if (_elapsedTimer) {
      clearInterval(_elapsedTimer);
      _elapsedTimer = null;
    }
  }

  function _updateElapsed() {
    if (!_generationStartTime) return;
    var elapsedMs = Date.now() - _generationStartTime;
    var elapsedSec = Math.floor(elapsedMs / 1000);
    $("#writing-elapsed-text").text(_formatDurationShort(elapsedSec));
    $("#writing-elapsed").removeClass("d-none");
  }

  function _formatDurationShort(seconds) {
    if (seconds < 60) return "<1m";
    var m = Math.floor(seconds / 60);
    if (m < 60) return m + "m";
    var h = Math.floor(m / 60);
    var rm = m % 60;
    return h + "h " + rm + "m";
  }

  function _updateTimeEstimate(current, total) {
    var remaining = total - current;

    // Update elapsed timer
    _updateElapsed();

    // Need at least 1 completed chapter to estimate
    if (_chapterCompletionTimes.length < 1 || remaining <= 0) {
      $("#writing-eta").addClass("d-none");
      return;
    }

    // Calculate average time per chapter from wall-clock data
    var startRef = _generationStartTime || _chapterCompletionTimes[0];
    var lastTime = _chapterCompletionTimes[_chapterCompletionTimes.length - 1];
    var elapsed = lastTime - startRef;
    var completed = _chapterCompletionTimes.length;
    var avgMs = elapsed / completed;

    var estRemainingMs = avgMs * remaining;
    var estSec = Math.round(estRemainingMs / 1000);

    var text = "~" + _formatDurationShort(estSec) + " remaining";

    // Also show avg time per chapter
    var avgMin = Math.round(avgMs / 60000);
    if (avgMin >= 1) {
      text += " (~" + avgMin + " min/ch)";
    }

    $("#writing-eta-text").text(text);
    $("#writing-eta").removeClass("d-none");
  }

  // -------------------------------------------------------------------
  // Step 4 – Done
  // -------------------------------------------------------------------
  function showDoneStep(data) {
    _doneData = data || {};
    _stopElapsedTimer();

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
      $("#editors-notes-empty").addClass("d-none");
      $("#consistency-assessment").text(consistency.overall_assessment || "");
      var $ul = $("#consistency-issues").empty();
      $.each(consistency.issues || [], function (_, issue) {
        $ul.append("<li>" + escapeHtml(issue) + "</li>");
      });
    } else {
      $("#consistency-alert").addClass("d-none");
      $("#editors-notes-empty").removeClass("d-none");
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
        '<span class="chapter-number-label">Chapter ' + escapeHtml(ch.number) + '</span>' +
        '<span class="chapter-title-label">' + escapeHtml(ch.title) + '</span>' +
        "</button></h2>" +
        '<div id="' + id + '" class="accordion-collapse collapse" aria-labelledby="' + heading + '">' +
        '<div class="accordion-body"><div class="chapter-reading-text"></div></div>' +
        "</div></div>"
      );
      // Use .text() to safely insert content with newlines preserved
      $item.find(".chapter-reading-text").text(ch.content || "");
      $acc.append($item);
    });

    // Writing statistics dashboard
    _renderWritingStats(data.chapters_done || []);

    // Character relationship map
    _renderRelationshipMap(data.character_relationship_map);

    // Populate the Writing tab timeline + progress bar so it isn't empty
    // when the user navigates back to it on a completed session.
    var doneCount = (data.chapters_done || []).length;
    var totalCh = data.total || doneCount || parseInt($("#chapters").val(), 10) || doneCount;
    _rebuildTimeline(data.chapters_done || [], totalCh);
    updateProgressBar(doneCount, totalCh, "Complete");
    if (doneCount > 0) {
      $("#writing-word-counter").removeClass("d-none");
    }

    showStep("#step-done");
    // Generation complete — re-enable the approve button for potential future use
    $("#btn-approve-outline").prop("disabled", false);
  }

  // Render Mermaid when the relationship-map tab becomes visible,
  // since Mermaid cannot measure layout inside a hidden tab pane.
  $(document).on("shown.bs.tab", "#pub-tab-relationships", function () {
    var $pre = $("#relationship-mermaid");
    var code = $pre.attr("data-mermaid-src");
    if (!code) return;
    try {
      $pre.removeAttr("data-processed").text(code);
      if (typeof mermaid !== "undefined" && mermaid.run) {
        mermaid.run({ nodes: [$pre[0]] });
      }
    } catch (e) { /* fallback: raw text already visible */ }
  });

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
    // Store source for deferred rendering when the tab becomes visible
    $pre.attr("data-mermaid-src", mermaidCode).text(mermaidCode);

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

    // Pre-compute per-chapter data and find max values
    var totalWords = 0, totalTime = 0, totalCalls = 0, totalTokens = 0;
    var totalErrors4xx = 0, totalErrors5xx = 0, totalTimeouts = 0;
    var maxWords = 0, maxTime = 0;
    var rows = [];

    $.each(chapters, function (_, ch) {
      var words = ch.word_count || (ch.content ? ch.content.split(/\s+/).length : 0);
      var timeSec = ch.generation_time_seconds || 0;
      var calls = ch.llm_calls || 0;
      var tokens = ch.total_tokens || 0;
      var err4 = ch.http_errors_4xx || 0;
      var err5 = ch.http_errors_5xx || 0;
      var errT = ch.timeout_errors || 0;
      totalWords += words; totalTime += timeSec; totalCalls += calls; totalTokens += tokens;
      totalErrors4xx += err4; totalErrors5xx += err5; totalTimeouts += errT;
      if (words > maxWords) maxWords = words;
      if (timeSec > maxTime) maxTime = timeSec;
      rows.push({ ch: ch, words: words, timeSec: timeSec, calls: calls, tokens: tokens, errors: err4 + err5 + errT });
    });

    $.each(rows, function (_, r) {
      var timeStr = r.timeSec > 0 ? _formatDuration(r.timeSec) : "-";
      var tokensStr = r.tokens > 0 ? r.tokens.toLocaleString() : "-";
      var callsStr = r.calls > 0 ? r.calls.toLocaleString() : "-";
      var wordsPct = maxWords > 0 ? Math.round((r.words / maxWords) * 100) : 0;
      var wordsBadge = (r.words === maxWords && chapters.length > 1) ? ' <span class="nf-stat-badge nf-stat-badge-words">longest</span>' : "";
      var timeBadge = (r.timeSec === maxTime && maxTime > 0 && chapters.length > 1) ? ' <span class="nf-stat-badge nf-stat-badge-time">slowest</span>' : "";

      var errorsStr = r.errors > 0 ? r.errors.toLocaleString() : "-";
      var errorsCls = r.errors > 0 ? ' class="text-end text-danger"' : ' class="text-end"';

      $tbody.append(
        "<tr>" +
        "<td>" + escapeHtml(r.ch.number) + "</td>" +
        "<td>" + escapeHtml(r.ch.title || "") + "</td>" +
        '<td class="text-end nf-sparkline" style="background-size: ' + wordsPct + '% 60%">' + r.words.toLocaleString() + wordsBadge + "</td>" +
        '<td class="text-end">' + timeStr + timeBadge + "</td>" +
        '<td class="text-end">' + callsStr + "</td>" +
        '<td class="text-end">' + tokensStr + "</td>" +
        "<td" + errorsCls + ">" + errorsStr + "</td>" +
        "</tr>"
      );
    });

    // Summary cards
    var avgWords = chapters.length > 0 ? Math.round(totalWords / chapters.length) : 0;
    var avgTimeSec = (chapters.length > 0 && totalTime > 0) ? Math.round(totalTime / chapters.length) : 0;
    var summaryItems = [
      { label: "Total Words", value: totalWords.toLocaleString(), icon: "bi-file-text" },
      { label: "Avg Words/Ch", value: avgWords.toLocaleString(), icon: "bi-calculator" },
    ];
    if (avgTimeSec > 0) {
      summaryItems.push({ label: "Avg Time/Ch", value: _formatDuration(avgTimeSec), icon: "bi-stopwatch" });
    }
    if (totalTime > 0) {
      summaryItems.push({ label: "Total Gen Time", value: _formatDuration(totalTime), icon: "bi-clock" });
    }
    if (totalCalls > 0) {
      summaryItems.push({ label: "LLM Calls", value: totalCalls.toLocaleString(), icon: "bi-chat-dots" });
    }
    if (totalTokens > 0) {
      summaryItems.push({ label: "Total Tokens", value: totalTokens.toLocaleString(), icon: "bi-cpu" });
    }
    var totalLLMErrors = totalErrors4xx + totalErrors5xx + totalTimeouts;
    if (totalLLMErrors > 0) {
      var breakdown = [];
      if (totalErrors4xx > 0) breakdown.push(totalErrors4xx + " 4xx");
      if (totalErrors5xx > 0) breakdown.push(totalErrors5xx + " 5xx");
      if (totalTimeouts > 0) breakdown.push(totalTimeouts + " timeout");
      summaryItems.push({ label: "LLM Errors", value: totalLLMErrors.toLocaleString(), icon: "bi-exclamation-triangle", cls: "text-danger", tooltip: breakdown.join(", ") });
    }

    $.each(summaryItems, function (_, item) {
      var iconCls = item.cls || "text-primary";
      var tooltipAttr = item.tooltip ? ' data-bs-toggle="tooltip" title="' + escapeHtml(item.tooltip) + '"' : "";
      $summary.append(
        '<div class="col-6 col-md-4 col-lg-2">' +
        '<div class="card text-center h-100"' + tooltipAttr + '>' +
        '<div class="card-body py-2 px-1">' +
        '<i class="bi ' + item.icon + ' ' + iconCls + ' mb-1 d-block"></i>' +
        '<div class="fw-bold ' + iconCls + '">' + item.value + "</div>" +
        '<small class="text-muted">' + item.label + "</small>" +
        "</div></div></div>"
      );
    });
    $summary.find('[data-bs-toggle="tooltip"]').each(function () {
      new bootstrap.Tooltip(this);
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

  $("#btn-rewrite-session-state").on("click", function () {
    clearAlerts();
    $("#rewrite-session-spinner").removeClass("d-none");
    var $btn = $("#btn-rewrite-session-state").prop("disabled", true);
    $.ajax({
      url: "/save_session_state",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({}),
      success: function () {
        showAlert("Session state saved to disk.", "success");
      },
      error: function (xhr) {
        var msg = (xhr.responseJSON && xhr.responseJSON.error) || "Failed to save session state.";
        showAlert(msg);
      },
      complete: function () {
        $("#rewrite-session-spinner").addClass("d-none");
        $btn.prop("disabled", false);
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
        '<div class="nf-empty-state"><i class="bi bi-palette"></i><p>Your illustrations will appear here</p></div>'
      );
    } else {
      // Sort: cover first, then by chapter number
      var sorted = illustrations.slice().sort(function (a, b) {
        if (a.type === "cover") return -1;
        if (b.type === "cover") return 1;
        return (a.chapter || 0) - (b.chapter || 0);
      });

      $.each(sorted, function (i, illust) {
        var isCover = illust.type === "cover";
        var label = isCover ? "Cover" : "Chapter " + (illust.chapter || "?");
        var safeUrl = escapeHtml(illust.image_url || "");
        var badgeHtml = isCover ? '<span class="nf-illust-badge">Cover</span>' : "";
        var coverClass = isCover ? " nf-illust-cover" : "";
        var desc = illust.scene_description || "";

        var $card = $(
          '<div class="nf-illust-card' + coverClass + '" data-illust-idx="' + i + '">' +
          '<img src="' + safeUrl + '" class="nf-illust-img" alt="' + escapeHtml(label) + '" loading="lazy">' +
          '<div class="nf-illust-info">' +
          '<div class="nf-illust-label"></div>' +
          '<p class="nf-illust-desc"></p>' +
          '</div></div>'
        );
        $card.find(".nf-illust-label").html(escapeHtml(label) + badgeHtml);
        $card.find(".nf-illust-desc").text(desc);
        // Store data for lightbox
        $card.data("lightbox", { url: illust.image_url || "", label: label, desc: desc });
        $gallery.append($card);
      });
    }

    $gallery.removeClass("d-none");
  }

  // Illustration lightbox
  $(document).on("click", ".nf-illust-card", function () {
    var data = $(this).data("lightbox");
    if (!data || !data.url) return;
    $("#nf-lightbox-img").attr("src", data.url).attr("alt", data.label);
    $("#nf-lightbox-label").text(data.label);
    $("#nf-lightbox-desc").text(data.desc);
    bootstrap.Modal.getOrCreateInstance(document.getElementById("nf-lightbox")).show();
  });

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
    // Extract a human-readable label from the action field or infer from content
    var label = entry.action || inferStatusFromRequestEntry(entry);

    if (entry.type === "request") {
      // Extract the user prompt text (skip system messages for display)
      var content = "";
      if (entry.payload && entry.payload.messages) {
        entry.payload.messages.forEach(function(msg) {
          if (msg.role === "user") {
            content += truncateText(msg.content, 2500);
          }
        });
      }
      // Fallback: if no user message, show system message
      if (!content && entry.payload && entry.payload.messages) {
        entry.payload.messages.forEach(function(msg) {
          if (msg.role === "system") {
            content += truncateText(msg.content, 500);
          }
        });
      }
      return {
        type: "request",
        header: label || "Request to LLM",
        content: content,
        timestamp: entry.timestamp
      };
    } else if (entry.type === "response") {
      var content = "";
      if (entry.response && entry.response.choices && entry.response.choices[0]) {
        var message = entry.response.choices[0].message;
        if (message && message.content) {
          content = truncateText(message.content, 2500);
        }
      }
      return {
        type: "response",
        header: label ? label + " — Response" : "LLM Response",
        content: content,
        timestamp: entry.timestamp
      };
    }
    return null;
  }

  var _logMessageCount = 0;

  function addLogMessage(formatted) {
    if (!formatted || !formatted.content) return;

    var messageClass = formatted.type === "request" ? "request" : "response";
    var searchText = (formatted.header + " " + formatted.content).toLowerCase();
    var html =
      '<div class="llm-message ' + messageClass + '" data-search="' + escapeHtml(searchText) + '">' +
        '<div class="llm-bubble">' +
          '<div class="llm-bubble-header">' + escapeHtml(formatted.header) + '</div>' +
          '<div class="llm-bubble-content">' + escapeHtml(formatted.content) + '</div>' +
          '<div class="llm-bubble-timestamp">' + escapeHtml(formatted.timestamp) + '</div>' +
        '</div>' +
      '</div>';

    $("#llm-chat-messages").append(html);
    _logMessageCount++;
    $("#log-count-badge").text(_logMessageCount);

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
    _logMessageCount = 0;
    $("#log-count-badge").text("0");
    $("#log-search").val("");
    _scheduleLogPoll();
    setStickyStatus(DEFAULT_STICKY_STATUS, { force: true });

    // Clear the server-side log file
    $.post("/clear_log").fail(function () {
      // Non-fatal — display was already cleared
    });
  });

  // Log search/filter
  $("#log-search").on("input", function () {
    var query = $(this).val().toLowerCase().trim();
    if (!query) {
      $("#llm-chat-messages .llm-message").show();
      return;
    }
    $("#llm-chat-messages .llm-message").each(function () {
      var searchData = $(this).attr("data-search") || "";
      $(this).toggle(searchData.indexOf(query) !== -1);
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

      // Hide hero when restoring a session
      $("#nf-hero").addClass("d-none");

      // Step 1 form fields
      if (sd.premise) $("#premise").val(sd.premise);
      if (sd.genre) $("#genre").val(sd.genre);
      if (sd.chapters) $("#chapters").val(sd.chapters);
      if (sd.word_count) $("#word_count").val(sd.word_count);
      if (sd.special_events) $("#special_events").val(sd.special_events);
      if (sd.special_instructions) $("#special_instructions").val(sd.special_instructions);
      if (sd.special_events || sd.special_instructions) {
        $("#advanced-options").addClass("show");
      }
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
          _startElapsedTimer();
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

  // -------------------------------------------------------------------
  // First-run tooltip tour (only on first visit, no restored session)
  // -------------------------------------------------------------------
  if (!window._savedSessionData && !localStorage.getItem("nf_tour_done")) {
    var tourSteps = [
      {
        target: "#premise",
        title: "Start here",
        content: "Describe the story you want to write — a premise, a mood, a world.",
        placement: "top",
      },
      {
        target: "#genre",
        title: "Set the tone",
        content: "Choose your genre to shape the style and feel of your novel.",
        placement: "top",
      },
    ];

    var _tourPopovers = [];

    function _showTourStep(idx) {
      if (idx >= tourSteps.length) {
        localStorage.setItem("nf_tour_done", "1");
        return;
      }
      var step = tourSteps[idx];
      var el = document.querySelector(step.target);
      if (!el) { _showTourStep(idx + 1); return; }

      var popover = new bootstrap.Popover(el, {
        title: step.title,
        content: step.content + '<div class="text-end mt-2"><button class="btn btn-sm btn-outline-secondary nf-tour-btn" data-tour-idx="' + idx + '">' +
          (idx < tourSteps.length - 1 ? "Next" : "Got it") + "</button></div>",
        placement: step.placement,
        trigger: "manual",
        html: true,
        customClass: "nf-tour-popover",
      });
      _tourPopovers.push(popover);
      popover.show();
    }

    $(document).on("click", ".nf-tour-btn", function () {
      var idx = parseInt($(this).attr("data-tour-idx"), 10);
      // Dispose current popover
      if (_tourPopovers[idx]) {
        _tourPopovers[idx].dispose();
      }
      _showTourStep(idx + 1);
    });

    // Dismiss all on any click outside a popover
    $(document).on("click", function (e) {
      if (!$(e.target).closest(".popover, .nf-tour-btn").length && _tourPopovers.length) {
        $.each(_tourPopovers, function (_, p) { try { p.dispose(); } catch (ex) {} });
        _tourPopovers = [];
        localStorage.setItem("nf_tour_done", "1");
      }
    });

    // Start tour after a short delay
    setTimeout(function () { _showTourStep(0); }, 800);
  }
});
