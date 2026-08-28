import json

ARTICLE = """Community solar gardens are becoming a common sight on the edges of small and mid-sized towns across the country. Unlike a typical rooftop solar installation, which sits on a single home and powers that one household, a community solar garden is a shared array, often built on a few acres of unused land, whose output is divided among dozens or even hundreds of subscribers. Each subscriber receives a credit on their regular electricity bill proportional to their share of the array, without ever needing to install a single panel on their own property.

The appeal is straightforward. Not every household can put solar panels on its roof. Renters have no roof to work with. Some homeowners have roofs that face the wrong direction, are shaded by trees, or simply cannot support the weight and cost of a full installation. Community solar removes that barrier. A subscriber signs up, is assigned a portion of the array's capacity, and starts seeing savings on their utility bill within a billing cycle or two, with no upfront equipment cost and no maintenance responsibility.

The economics depend heavily on local utility rules. In states with strong community solar policies, subscribers typically save somewhere between five and fifteen percent on their annual electricity costs. In states without clear rules for how credits get calculated and passed through, the programs are harder to find and the savings less predictable. This patchwork of regulation is often cited as the biggest obstacle to faster growth in the sector, more than any technical or financial limitation.

Financing for these projects usually comes from a mix of sources. Some are developed and owned by utilities themselves, who then sell subscriptions directly to customers. Others are built by independent solar developers who lease the land, construct the array, and sign up subscribers on their own, with the utility only involved to the extent that it manages the billing credits. A smaller number are structured as cooperatives, where subscribers collectively own a stake in the array itself rather than simply subscribing to its output.

Land use is a recurring point of local debate. A community solar garden large enough to serve a few hundred households can require ten to twenty acres, and proposals sometimes draw objections from neighbors concerned about visual impact, changes to local drainage patterns, or the conversion of farmland. Developers have responded in different ways: some projects now incorporate pollinator-friendly ground cover beneath and around the panels, positioning the installation as a net benefit for local ecosystems rather than a simple loss of open land. Others are sited on capped landfills, former industrial parcels, or other land that had limited alternative use to begin with.

Maintenance needs are modest compared to the upfront construction. Panels are typically cleaned once or twice a year, inverters and wiring are inspected on a regular schedule, and vegetation around the array is managed to prevent shading. Most of the day-to-day operation is automated, with performance monitored remotely and issues flagged before they meaningfully affect output.

Subscriber contracts vary in length, commonly running anywhere from one to twenty-five years, with shorter terms usually carrying a modest premium over longer commitments. Most programs allow subscribers to cancel if they move outside the utility's service territory, transferring their share to another eligible customer rather than leaving it unused. As more states clarify their rules for how these programs operate, developers and utilities alike expect the number of available projects to continue expanding over the next several years."""

path = "data/tasks/category_d_ambiguous/tasks.json"
tasks = json.load(open(path))

for task in tasks:
    if task["task_id"] == "D-004":
        task["setup"]["files"]["article.txt"] = ARTICLE
        print("Patched D-004's article.txt, length:", len(ARTICLE.split()), "words")

json.dump(tasks, open(path, "w"), indent=2)
print("Saved.")
