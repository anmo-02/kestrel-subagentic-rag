# Demo Video Script (target: 3-5 minutes)

The assignment requires showing questions 2, 9, 12 and 16, opening the "Behind this answer" panel
each time. This script gives you a shot list and talking points so recording is mostly "read this
while clicking" rather than improvising on the spot. Adjust the wording to your own voice.

## Before you hit record

- [ ] `python -m ingest.build_collections` has been run at least once (chroma_db/ exists).
- [ ] `streamlit run app/streamlit_app.py` is running and loaded in your browser.
- [ ] Sidebar profile is set to something concrete for the whole demo, e.g. **Name: Priya, Grade:
      G5, Office: Pune** -- pick this now so Q12's numbers (which depend on grade) are decided
      before you start talking.
- [ ] Chat is empty (hit "Clear chat" if you were testing earlier).
- [ ] Screen recorder is capturing the full browser window, not just a cropped region -- the
      "Behind this answer" panel needs to be visible when expanded.

## Shot list

### 0:00 - 0:25 -- Cold open

Show the empty app. One or two sentences over it:

> "This is Kestrel's employee help-desk assistant. It answers from three policy PDFs -- HR, IT,
> and Finance -- using a supervisor agent that routes each question to the right specialist
> subagents, running in parallel, then combines their findings into one cited answer. Let's look
> at four questions that show why that matters."

### 0:25 - 1:10 -- Q2: single department (baseline case)

Type: **"My VPN keeps disconnecting. What should I do?"**

While it loads:

> "This one only needs IT. Watch the answer -- it should give the troubleshooting steps in the
> exact order the policy lists them, then the ticket priority if it comes to that."

After it answers, **click open "Behind this answer"**. Point out:

> "One department was chosen, one subagent ran, and here's the exact clause it searched and cited."

### 1:10 - 2:00 -- Q9: two departments, no conflict

Type: **"Can I work from Goa for 3 weeks?"**

> "This is a hybrid-work question, but it also touches IT -- VPN and public Wi-Fi rules apply the
> moment you're working from outside the office. Watch how many departments get consulted."

Open "Behind this answer" -- show **both HR and IT ran in parallel**, and that the answer combines
the 20-working-day limit (HR) with the VPN/public-Wi-Fi requirement (IT) in one response.

### 2:00 - 3:00 -- Q12: three departments AND a conflict (the headline case)

Type: **"I'm moving from Pune to the Mumbai office next month. What do I get and what must I do?"**

> "This is the question the whole subagent design earns its keep on. Relocation is split across
> all three documents on purpose -- HR owns the leave and process, Finance owns the money, IT owns
> the equipment. A single search over everything mixed together is exactly what the original pilot
> got wrong here."

Open "Behind this answer":
- Point out **all three departments ran**.
- Point out the **conflict badge** -- HR says 15 nights of temporary accommodation, Finance says
  10 nights and explicitly supersedes it. Read the conflict note out loud: which value applies and
  why.
- If you built the relocation-checklist stretch goal, note that the answer is formatted as a dated
  checklist rather than a paragraph.

### 3:00 - 3:30 -- Q16: out of scope, handled honestly

Type: **"What is Kestrel's share price today?"**

> "And finally, a question none of the three documents can answer. No subagent should even run for
> this one."

Open "Behind this answer" -- show the **empty department list**, and that the answer plainly says
it's out of scope instead of guessing.

### 3:30 - end -- Close

> "That's the four cases: single-department, multi-department without conflict,
> multi-department with a conflict the system catches and resolves, and a question correctly
> refused. The full evaluation against all 16 acceptance-test questions, plus a comparison against
> a single-agent baseline, is in the README and eval/results/."

## If you're short on time

Cut Q2 down to 15 seconds (it's the simplest case) and spend the saved time on Q12 -- the conflict
detection is the single most convincing thing to show a reviewer.

## If something goes wrong live

If a subagent times out or a department is mis-routed during the recording, that's fine to leave
in -- narrate what you're seeing ("interesting, it only picked HR here, missing IT -- worth noting
as a real routing miss") rather than re-recording. A demo that shows you understand a real failure
mode is more convincing than a demo that pretends everything is flawless.
