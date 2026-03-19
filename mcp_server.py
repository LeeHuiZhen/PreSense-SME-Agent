# mcp_server.py
from fastmcp import FastMCP
import pandas as pd

mcp = FastMCP("PreSense_Maintenance_Server")

@mcp.tool()
def check_inventory(part_name: str) -> str:
    """Check inventory for required spare parts by name or number."""
    try:
        inv = pd.read_csv('inventory.csv')
        
        available = inv[
            (inv['stock'] > 0) &
            (
                inv['part_name'].str.contains(part_name, case=False, na=False) |
                inv['part_number'].str.contains(part_name, case=False, na=False)
            )
        ]
        
        if not available.empty:
            part = available.iloc[0]
            return (f"PARTS LOCATED: {part['stock']} units of "
                    f"{part['part_name']} @ {part['location']}. "
                    f"Cost: RM {part['unit_cost_myr']}")
                    
        return f"OUT OF STOCK: {part_name} not available."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def schedule_technician(certification: str) -> str:
    """Schedule an available technician by certification."""
    try:
        sch = pd.read_csv('schedule.csv')
        available = sch[
            (sch['status'] == 'Available') &
            (sch['certification'].str.contains(certification, case=False, na=False))
        ]
        if not available.empty:
            tech = available.iloc[0]
            return (f"ASSIGNED: {tech['name']} "
                    f"({tech['contact']}). ETA: 2 hours.")
        return "NO AVAILABLE TECHNICIANS with that certification."
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def create_work_order(unit_id: int, cycle: int, predicted_rul: float, decision: str = "AUTO_DISPATCH") -> str:
    """Create and dispatch a maintenance work order."""
    wo_id  = f"WO-{unit_id:03d}-{cycle:04d}"
    status = "DISPATCHED" if decision == "AUTO_DISPATCH" else "RESERVED"
    action = "AUTO-DISPATCHED" if decision == "AUTO_DISPATCH" else "SCHEDULED"
    return (f"WORK ORDER {wo_id} {action}. "
            f"Unit #{unit_id:03d}, Cycle {cycle}, "
            f"RUL {predicted_rul:.0f} cycles. STATUS: {status}.")

if __name__ == "__main__":
    mcp.run()