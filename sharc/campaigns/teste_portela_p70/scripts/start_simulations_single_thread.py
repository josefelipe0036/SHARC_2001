from sharc.run_multiple_campaigns import run_campaign

# Set the campaign name to our new p=70 campaign folder
name_campaign = "teste_portela_p70"

# Run the campaign in single-thread mode
print(f"Starting campaign: {name_campaign}...")
run_campaign(name_campaign)
print("Campaign execution completed!")
