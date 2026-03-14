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

  function showStep(id) {
    $.each(STEPS, function (_, sel) {
      $(sel).addClass("d-none");
    });
    $(id).removeClass("d-none");
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
  // Premise character counter
  // -------------------------------------------------------------------
  $("#premise").on("input", function () {
    var len = $(this).val().length;
    $("#premise-count").text(len);
    if (len > 1200) {
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
    if (!premise || premise.length > 1200) {
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
      var row =
        "<tr>" +
        "<td>" + escapeHtml(ch.number || "") + "</td>" +
        "<td><div class='editable-cell' contenteditable='true' data-field='title'>" +
        escapeHtml(ch.title || "") +
        "</div></td>" +
        "<td><div class='editable-cell' contenteditable='true' data-field='summary'>" +
        escapeHtml(ch.summary || "") +
        "</div></td>" +
        "</tr>";
      $tbody.append(row);
    });

    // Characters table
    var $ctbody = $("#characters-tbody").empty();
    $.each(data.characters || [], function (_, c) {
      var row =
        "<tr>" +
        "<td><div class='editable-cell' contenteditable='true' data-field='name'>" + escapeHtml(c.name || "") + "</div></td>" +
        "<td><div class='editable-cell' contenteditable='true' data-field='age'>" + escapeHtml(c.age || "") + "</div></td>" +
        "<td><div class='editable-cell' contenteditable='true' data-field='role'>" + escapeHtml(c.role || "") + "</div></td>" +
        "<td><div class='editable-cell' contenteditable='true' data-field='background'>" + escapeHtml(c.background || "") + "</div></td>" +
        "<td><div class='editable-cell' contenteditable='true' data-field='arc'>" + escapeHtml(c.arc || "") + "</div></td>" +
        "</tr>";
      $ctbody.append(row);
    });
  }

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
  var _pollInterval = null;
  var _progressToken = null;
  var _totalChapters = 0;

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
    }

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

  // -------------------------------------------------------------------
  // LLM Log Display
  // -------------------------------------------------------------------
  var _lastLogLength = 0;
  var _logPollInterval = null;

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
        if (data.entries && data.entries.length > _lastLogLength) {
          // Clear placeholder if first time
          if (_lastLogLength === 0) {
            $("#llm-chat-messages").empty();
          }
          
          // Add new entries
          for (var i = _lastLogLength; i < data.entries.length; i++) {
            var formatted = formatLogEntry(data.entries[i]);
            if (formatted) {
              addLogMessage(formatted);
            }
          }
          _lastLogLength = data.entries.length;
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
    _lastLogLength = 0;
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
    $("#chapters-preview-accordion").empty();
    $("#chapter-progress-list").empty();

    showStep("#step-input");
  });
});
