# In-Dome Considerations

This subdirectory archives all research, analysis, and planning related to the **final telescope dome deployment** of the BATPING acoustic temperature measurement system.

## Scope

Documentation here covers issues specifically related to the in-dome environment and final implementation:

- **Signal processing and optimization** for dome acoustic environment
- **Background characterization and noise reduction** strategies
- **Hardware concerns and planning** for dome deployment
- **Implementation strategies** for the final apparatus

This is distinct from general development/prototyping work in the parent `agilent_project/` directory.

## Contents

| File | Description |
|------|-------------|
| `README.md` | This file |
| `research_log.md` | Detailed research notebook with conversation excerpts |
| `plots/` | Noise characterization plots and spectrum data |

## Project Context

BATPING is an acoustic temperature measurement system for telescope dome seeing prediction. It uses ultrasonic time-of-flight measurements with a 40kHz piezoelectric transducer array. The speed of sound depends on air temperature, enabling precise temperature inference from pulse timing.

Key challenges for dome deployment:
1. Environmental ultrasonic interference (HVAC, motors, etc.)
2. In-band vs out-of-band noise mitigation strategies
3. Transducer resonant behavior and pulse optimization

See parent directory and `/batping/` for development history.

## Related Files

- `../agilent_33220a.py` - Agilent 33220A function generator interface
- `../pulse_optimizer.py` - Pulse optimization framework
- `/batping/WORKLOG.md` - Earlier development notes
- `/batping_process_explanation.txt` - System overview

---
*This subdirectory will be updated as hardware concerns and implementation ideas develop over Summer 2026.*
