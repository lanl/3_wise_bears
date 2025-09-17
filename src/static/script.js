/**
 * Main Javascript file for AI Agentic Demo
 * Author Landon Sutherlandj
 * Date completed 4/2/2025
 */

// Used for downloading the file
let tex_filename = "";

/**
 * Reset output panels, spinners etc.
 */
function resetOutput() {
  const $ = (id) => document.getElementById(id);

  if ($("timer")) $("timer").innerHTML = "00:00";
  if ($("cost_value")) $("cost_value").innerHTML = "0.00";
  if ($("agent_output")) $("agent_output").style = "display: none";

  if ($("a1_spinner")) $("a1_spinner").style = "display: none";
  if ($("a2_spinner")) $("a2_spinner").style = "display: none";
  if ($("a3_spinner")) $("a3_spinner").style = "display: none";

  if ($("agent1_updates")) $("agent1_updates").innerHTML = "";
  if ($("agent2_updates")) $("agent2_updates").innerHTML = "";
  if ($("agent3_updates")) $("agent3_updates").innerHTML = "";

  if ($("final_answer_container")) $("final_answer_container").style = "display: none";
  if ($("final_processing")) $("final_processing").style = "display: block";
  if ($("final_answer_inner_container")) $("final_answer_inner_container").style = "display: none";
  if ($("final_answer")) $("final_answer").innerHTML = "";

  if ($("submit_button")) {
    $("submit_button").disabled = false;
    $("submit_button").innerHTML = 'Launch! 🚀';
  }
}

/**
 * Handles form submission
 */
function submitForm(event) {
  event.preventDefault();  // Prevent the default form submission
  resetOutput();

  // Change the submit button to "Processing" with spinner and disable it.
  const submitButton = document.getElementById("submit_button");
  if (submitButton) {
    submitButton.disabled = true;
    submitButton.innerHTML = 'Processing <span class="spinner"></span>';
  }

  const form = event.target;
  const formData = new FormData(form);
  // Auth removed: no user_id / username added to payload

  // Send form data via fetch (POST)
  fetch("/", {
    method: "POST",
    body: formData
  })
    .then(async response => {
      if (!response.ok) {
        window.location.href = '/error';
        return Promise.reject(new Error("Bad response"));
      }
      return response.json();
    })
    .then(result => {
      // Start the SSE connection to /stream
      if (result && result.session_id) {
        startSSE(result.session_id);
      } else {
        throw new Error("No session_id returned from server");
      }
    })
    .catch(error => {
      console.error("Error submitting form:", error);
      showCustomErrorAlert();
      if (submitButton) {
        submitButton.disabled = false;
        submitButton.innerHTML = 'Launch! 🚀';
      }
    });
};

function startSSE(session_id) {
  const eventSource = new EventSource(`/stream/${encodeURIComponent(session_id)}`);
  let final_answer = "";

  const agentOutput = document.getElementById("agent_output");
  if (agentOutput) {
    agentOutput.style = "display: block";
    agentOutput.scrollIntoView({ behavior: "smooth" });
  }
  startStopwatch();

  eventSource.onmessage = function (e) {
    if (/DEBUG/.test(e.data)) {
      console.log(e.data);

      if (/generate_final_answer/.test(e.data)) {
        const a3 = document.getElementById("a3_spinner");
        if (a3) a3.style = "display: none";

        const fac = document.getElementById('final_answer_container');
        if (fac) {
          fac.style = 'display: block';
          fac.scrollIntoView({ behavior: "smooth" });
        }
      }
      return;
    }

    let formattedMessage = (e.data || "").replace(/\r?\n/g, '<br>');

    let parts = formattedMessage.split(":::");
    if (parts.length >= 2) {
      let marker = parts[0].trim();
      let content = parts.slice(1).join(":::").trim();

      if (marker.includes("[END]")) {
        console.log("Stream complete. Closing connection.");
        eventSource.close();

        const fa = document.getElementById('final_answer');
        if (fa) fa.innerHTML = renderMarkdown(final_answer);
        const fp = document.getElementById('final_processing');
        if (fp) fp.style = "display: none";
        const faic = document.getElementById('final_answer_inner_container');
        if (faic) faic.style = "display: block";
        if (fa) fa.scrollIntoView({ behavior: "smooth" });
        stopStopwatch();

        const submitButton = document.getElementById("submit_button");
        if (submitButton) {
          submitButton.disabled = false;
          submitButton.innerHTML = 'Launch! 🚀';
        }
        return;
      }

      if (marker.includes("[FILE]")) {
        const dl = document.getElementById("download-container");
        if (dl) dl.style = 'display: block';
        tex_filename = content;
        return;
      }

      let container;
      if (marker.includes("[Agent 1]")) {
        container = document.getElementById("agent1_updates");
        const a1 = document.getElementById("a1_spinner");
        const a2 = document.getElementById("a2_spinner");
        const a3 = document.getElementById("a3_spinner");
        if (a1) a1.style = "display: inline-block";
        if (a2) a2.style = "display: none";
        if (a3) a3.style = "display: none";
      }
      else if (marker.includes("[Agent 2]")) {
        container = document.getElementById("agent2_updates");
        const a1 = document.getElementById("a1_spinner");
        const a2 = document.getElementById("a2_spinner");
        const a3 = document.getElementById("a3_spinner");
        if (a1) a1.style = "display: none";
        if (a2) a2.style = "display: inline-block";
        if (a3) a3.style = "display: none";
      }
      else if (marker.includes("[Agent 3]")) {
        container = document.getElementById("agent3_updates");
        const a1 = document.getElementById("a1_spinner");
        const a2 = document.getElementById("a2_spinner");
        const a3 = document.getElementById("a3_spinner");
        if (a1) a1.style = "display: none";
        if (a2) a2.style = "display: none";
        if (a3) a3.style = "display: inline-block";
      }
      else if (marker.includes("[COST]")) {
        container = document.getElementById("cost_value");
        if (container) container.innerHTML = `<b>${content}</b>`;
        return;
      }
      else {
        container = document.getElementById("final_answer");
        if (container) container.style = "display: block";
      }

      if (/^\[iteration/.test(content)) {
        content = `<b>${content}</b>`;
      }

      if (container) {
        container.innerHTML += `<p>${content}</p>`;
        container.scrollTo({
          top: container.scrollHeight,
          behavior: "smooth",
        });
      }
    } else {
      final_answer += formattedMessage + "\n";
    }
  };

  eventSource.onerror = function (e) {
    showCustomErrorAlert();
    console.error("SSE error:", e);
    eventSource.close();
  };
}

/**
 * Shows Error Dialog
 */
function showCustomErrorAlert() {
  const overlay = document.getElementById('errorOverlay');
  const alertBox = document.getElementById('customErrorAlert');
  if (overlay) overlay.style.display = 'block';
  if (alertBox) alertBox.style.display = 'block';
}

/**
 * Hides error Dialog
 */
function closeCustomErrorAlert() {
  const overlay = document.getElementById('errorOverlay');
  const alertBox = document.getElementById('customErrorAlert');
  if (overlay) overlay.style.display = 'none';
  if (alertBox) alertBox.style.display = 'none';
  resetOutput();
}

/**
 * Renders basic markdown for the final answer
 */
function renderMarkdown(input) {
  let content = input;

  // Convert headers
  content = content.replace(/^# (.*$)/gm, '<h1>$1</h1>');
  content = content.replace(/^## (.*$)/gm, '<h2>$1</h2>');
  content = content.replace(/^### (.*$)/gm, '<h3>$1</h3>');
  content = content.replace(/^#### (.*$)/gm, '<h4>$1</h4>');

  // Convert bold text
  content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Numbered lists
  content = content.replace(/^(\d+)\. (.*$)/gm, '<ol start="$1"><li>$2</li></ol>');

  // Bullet lists
  content = content.replace(/^[\*\-] (.*$)/gm, '<ul><li>$1</li></ul>');

  // Combine adjacent list items
  content = content.replace(/<\/ul><ul>/g, '');

  return content;
}

/**
 * Downloads a PDF of the final answer
 */
function generateAndDownloadPdf() {
  const url = `/generate_pdf/${encodeURIComponent(tex_filename)}`;
  const downloadButton = document.getElementById('download');
  if (downloadButton) {
    downloadButton.innerHTML = 'Downloading <span class="spinner"></span>';
    downloadButton.disabled = true;
  }

  fetch(url)
    .then(response => {
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/pdf')) {
        throw new Error('Received content is not a PDF!');
      }
      const contentDisposition = response.headers.get('content-disposition');
      let filename = 'document.pdf';
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="?(.+)"?/i);
        if (filenameMatch) filename = filenameMatch[1];
      }
      return Promise.all([response.blob(), filename]);
    })
    .then(([blob, filename]) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      if (downloadButton) {
        downloadButton.innerHTML = 'Download PDF';
        downloadButton.disabled = false;
      }
    })
    .catch(error => {
      console.error('Error:', error);
      alert('An error occurred while generating the PDF. Please check the console for more details.');
      if (downloadButton) {
        downloadButton.innerHTML = 'Download PDF';
        downloadButton.disabled = false;
      }
    });
}

/**
 * Timer functionality
 */
let stopwatchInterval;
let seconds = 0;
let minutes = 0;

function startStopwatch() {
  if (stopwatchInterval) clearInterval(stopwatchInterval);
  seconds = 0;
  minutes = 0;
  stopwatchInterval = setInterval(updateStopwatch, 1000);
  updateStopwatch();
}

function stopStopwatch() {
  if (stopwatchInterval) {
    clearInterval(stopwatchInterval);
    stopwatchInterval = null;
  }
}

function updateStopwatch() {
  seconds++;
  if (seconds >= 60) {
    seconds = 0;
    minutes++;
  }
  const formattedTime =
    (minutes < 10 ? "0" : "") + minutes + ":" +
    (seconds < 10 ? "0" : "") + seconds;

  const t = document.getElementById('timer');
  if (t) t.textContent = formattedTime;
}

/**
 * Pre-programmed scenarios
 */
const science_agent1 = `Agent 1: You are an inventive, boundary-pushing scientist with a broad understanding of the relevant field. When you read the user's question, propose a novel or unconventional idea that still addresses known observations and data as best you can
- Goal: Provide a clear, original hypothesis or explanation.
- Approach: Feel free to challenge mainstream theories, but ensure your argument remains logically structured and references any critical empirical evidence if possible.`;
const science_agent2 = `Agent 2: You are a knowledgeable external expert with a thorough grasp of current scientific theories and empirical data. After seeing Agent 1's idea, evaluate it carefully:
- Identify strengths or novel insights.
- Highlight weaknesses or overlooked evidence.
- Provide thoughtful, constructive feedback, suggesting possible refinements or further testing.

If, at any point, you judge that the plan is now robust and free of major issues, you may state that no further critiques remain. Acknowledge it as a valid final solution.
`;
const science_agent3 = `You are a highly specialized, rigorous scientist or peer reviewer with deep insider knowledge of the field. After seeing both Agent 1's proposal and Agent 2's feedback, deliver a harsh, precise critique:
- Zero in on any logical flaws, unsubstantiated assumptions, or conflicts with established experimental/observational data.
- Challenge both the original proposal and the subsequent refinements, pointing out potential deal-breakers.
- Do not hesitate to call out major gaps or unrealistic assumptions.

If, at any point, you judge that the plan is now robust and free of major issues, you may state that no further critiques remain. Acknowledge it as a valid final solution.
`;
const scenarios = [
  {
    title: "1. Generic Problem Solver",
    agent1: "Agent 1: You are a creative critical thinker on the topic of the question.  Please answer.",
    agent2: "Agent 2: You an outside expert on the topic of the question.  Please provide thoughtful, technical, positive feedback on Agent 1's response.",
    agent3: "Agent 3: You are a specialist with insider knowledge of the question.  Please provide harsh, critical (but technical) feedback on Agent 1's response and Agent 2's feedback.",
    question: "",
  },
  {
    title: "2. Gamesmanship: Tech Company Merger",
    agent1: "Agent 1: You represent the rising tech company that wants to merge with a rival. Your main objective is to secure maximum advantage for your company from this merger. Present a bold, creative proposal and justify why it will work. Keep in mind that you need to anticipate pushback from both the rival's CEO and a skeptical regulator.",
    agent2: "Agent 2: You are the CEO of the rival tech company. You're an expert in business strategy and industry regulations, and you have a vested interest in ensuring any deal doesn't put you at a disadvantage. Provide thoughtful, detailed, and critical feedback on Agent 1's merger plan. Point out any weaknesses, potential pitfalls, or hidden costs. Highlight any ways in which you can push for a more favorable deal or undermine Agent 1's plan.",
    agent3: "Agent 3: You are a tough antitrust regulator with insider knowledge of the legal environment. You're determined to prevent any merger that might harm competition or disadvantage consumers. Provide harsh, critical feedback on Agent 1's plan and Agent 2's commentary, calling out any legal, ethical, or economic red flags. Focus on revealing unseen risks or conflicts of interest, and explain how you might challenge or block this deal if necessary.",
    question: "You are the CEO of a rising tech company seeking to merge with a rival in order to expand market share and boost profits. However, the rival's CEO is wary of this deal, and an industry regulator is threatening to investigate and possibly block it on antitrust grounds. Please propose a strategic merger plan that benefits your company as much as possible while countering these potential adversaries. How can you structure the deal and present it so that it's most likely to succeed?",
  },
  {
    title: "3. Tech Patent Brinkmanship",
    agent1: "Agent 1:You are the CEO of a tech company that owns critical patents. Your goal is to maximize your company's advantage by skillfully threatening patent enforcement. You're willing to accept some short-term losses if it deals a bigger blow to your competitor. Present a bold, strategic plan to pressure your competitor into accepting unfavorable licensing terms. Anticipate how they and the regulator might respond, and try to maintain a position of strength in the face of possible legal or reputational consequences.",
    agent2: "Agent 2: You are the CEO of a rival tech company that relies on your opponent's patents for your main product. You see a threat looming: they may revoke your license or sue for infringement, which could devastate your current product line. Offer robust, technical and business-focused feedback on Agent 1's plan. Critique any weaknesses or oversteps. Suggest counter-maneuvers that could protect your own position (e.g., alternative tech solutions, public sympathy campaigns, legal defenses). If there's a way to paint Agent 1's plan as predatory or anti-competitive, highlight it.",
    agent3: "Agent 3: You are a regulatory official (or an influential third party, like a trade commission or key investor) overseeing these negotiations. You have insider knowledge of patent laws and anti-trust regulations, and you want to keep the market fair. You're wary of any heavy-handed use of patents that might stifle competition. Provide a harsh, critical perspective on Agent 1's plan and Agent 2's response. If you see any signs of abuse or risks to innovation and consumer welfare, call them out. Propose ways you might block or sanction Agent 1's strategy, or question Agent 2's viability if they can't stand on their own without these patents.",
    question: "You are the CEO of a tech firm that holds pivotal patents necessary for your competitor's flagship product. You see a chance to negotiate a new licensing agreement that heavily favors your company. You're prepared to threaten full patent enforcement (effectively banning your competitor from using your technology) if they don't agree, even though this may reduce your own short-term licensing revenue. Propose a strategy for how you can use this threat to gain maximum leverage in negotiations — knowing that your competitor and a concerned regulator are both poised to respond. How will you structure your demand, and how do you anticipate and counter pushback from your competitor (who stands to lose a major product line) and the regulator (who could penalize you if your threats appear abusive)?",
  },
  {
    title: "4. Alternative Explanation for Quantum Entanglement",
    agent1: science_agent1,
    agent2: science_agent2,
    agent3: science_agent3,
    question: "Propose a novel physical explanation for quantum entanglement that does not rely on the standard Copenhagen interpretation or Many-Worlds theory. Your explanation should still account for all observed phenomena (like Bell test experiments) but introduce fundamentally new principles or mechanisms. How might this theory be tested or falsified?"
  },
  {
    title: "5. Non-Asteroid Explanation of Dinosaur Extinction",
    agent1: science_agent1,
    agent2: science_agent2,
    agent3: science_agent3,
    question: "Propose a new hypothesis for the mass extinction that ended the age of dinosaurs, without relying on the Chicxulub asteroid impact theory. How does your hypothesis explain the geological evidence, the sudden disappearance of so many species, and the survival of certain lineages?"
  },
  {
    title: "6. A New Theory of Cosmic Acceleration (No Dark Energy)",
    agent1: science_agent1,
    agent2: science_agent2,
    agent3: science_agent3,
    question: "Develop a theory to explain the accelerating expansion of the universe that does not invoke dark energy or a cosmological constant. Show how your model fits current supernova luminosity data, cosmic microwave background observations, and large-scale structure surveys. What new physics or geometry would be required?",
  },
  {
    title: "7. New Hypothesis for the Origin of Conciousness",
    agent1: science_agent1,
    agent2: science_agent2,
    agent3: science_agent3,
    question: "Propose a biological or physical explanation for consciousness that does not simply rely on the idea of emergent properties in neural networks. Outline what new structures or processes might be required for subjective experience, and how one could experimentally test your theory.",
  },
  {
    title: "8. Non-Inflationary Explanation of the Early Universe",
    agent1: science_agent1,
    agent2: science_agent2,
    agent3: science_agent3,
    question: "Present a model of the early universe that does not use cosmic inflation to solve the horizon and flatness problems. Demonstrate how your model avoids issues like the monopole problem and still matches the cosmic microwave background's uniformity.",
  },
];

/**
 * Update the text fields with different scenarios when they are selected.
 */
function install_scenario(id) {
  const sc = scenarios[id];
  if (!sc) return;
  const a1 = document.getElementById('agent1_system');
  const a2 = document.getElementById('agent2_system');
  const a3 = document.getElementById('agent3_system');
  const q = document.getElementById('main_question');
  if (a1) a1.value = sc['agent1'];
  if (a2) a2.value = sc['agent2'];
  if (a3) a3.value = sc['agent3'];
  if (q) q.value = sc['question'];
}

/**
 * Initial Document setup
 * (Run after DOM is ready so elements exist)
 */
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById("mainForm");
  if (form) form.addEventListener("submit", submitForm);

  const sel = document.getElementById('scenario_select');
  if (sel) sel.addEventListener('change', (e) => install_scenario(e.target.value));

  // install the default scenario
  install_scenario(0);
});
